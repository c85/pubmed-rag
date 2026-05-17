import json
import time
import pyodbc
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import ollama
from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from prefect import runtime
from azure.storage.blob import BlobServiceClient
from prefect_azure import AzureBlobStorageCredentials

# ── Constants ────────────────────────────────────────────────────────────────
BLOB_RAW       = "pubmed-raw"
BLOB_PROCESSED = "pubmed-processed"
OLLAMA_MODEL   = "qwen2.5-coder:32b"
PROMPT_VERSION = "v1.0"
MAX_WORKERS    = 2

# Sections not worth embedding for a clinical RAG system
DISCARD_TYPES  = {"methods", "funding", "acknowledgements", "discard"}

# ── Prompts ───────────────────────────────────────────────────────────────────
CLASSIFY_PROMPT = """\
You are a medical literature parser. Given a section title and a short content \
preview from a pediatric oncology research article, classify the section into \
exactly one of these categories:

  abstract | background | intervention | results | discussion | conclusions | methods | discard

Use "discard" for: acknowledgements, funding, author contributions, \
conflict of interest, references, or appendices.

Return ONLY valid JSON with no other text:
{{"section_type": "<category>"}}

Section title: {title}
Content preview: {preview}
"""

EXTRACT_PROMPT = """\
You are a medical literature parser. Extract structured metadata from this \
section of a pediatric oncology research article.

Return ONLY valid JSON with no other text:
{{
  "cancer_type":  ["list of cancer types mentioned, e.g. ALL, AML, neuroblastoma"],
  "treatments":   ["list of treatments, drugs, or interventions mentioned"],
  "study_design": "one of: RCT, observational, retrospective, prospective, \
meta-analysis, review, case-series, other, unknown",
  "phase":        "clinical trial phase if mentioned: I, II, III, IV — or null",
  "sample_size":  <integer or null>
}}

Section text:
{text}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ollama_json(prompt: str, retries: int = 3) -> Optional[dict]:
    """Call Ollama and return parsed JSON, or None on repeated failure."""
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return json.loads(response["message"]["content"])
        except (json.JSONDecodeError, KeyError, ollama.ResponseError) as e:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)  # exponential backoff
    return None


def _section_text(sec_el: ET.Element) -> str:
    """Extract all text from a <sec> element, stripping XML tags."""
    return " ".join(sec_el.itertext()).strip()


def _get_conn(conn_str: str) -> pyodbc.Connection:
    return pyodbc.connect(conn_str, timeout=30)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="get_processed_pmids")
def get_processed_pmids(blob_creds: AzureBlobStorageCredentials) -> set[str]:
    """Return PMIDs already present in the processed container (idempotency)."""
    conn_str  = blob_creds.connection_string.get_secret_value()
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client(BLOB_PROCESSED)
    if not container.exists():
        return set()
    return {b["name"].removesuffix(".json") for b in container.list_blobs()}


@task(name="download_raw", retries=3, retry_delay_seconds=10)
def download_raw(pmid: str, blob_creds: AzureBlobStorageCredentials) -> str:
    """Download raw JATS XML for a single PMID from Azure Blob."""
    conn_str  = blob_creds.connection_string.get_secret_value()
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client(BLOB_RAW)
    blob      = container.download_blob(f"{pmid}.xml")
    return blob.readall().decode("utf-8")


@task(name="parse_jats")
def parse_jats(pmid: str, xml_str: str) -> dict:
    """
    Parse JATS XML into a structured dict with bibliographic metadata
    and a list of raw sections (title + text) ready for LLM classification.
    """
    logger = get_run_logger()
    root   = ET.fromstring(xml_str)
    front  = root.find("./front/article-meta")

    # ── Bibliographic metadata from <article-meta> ──────────────────────────
    year_el   = front.find(".//pub-date/year") if front is not None else None
    journal_el = root.find("./front/journal-meta/journal-title-group/journal-title")

    meta = {
        "pmid":         pmid,
        "year":         int(year_el.text) if year_el is not None and year_el.text else None,
        "journal":      journal_el.text.strip() if journal_el is not None else None,
        "article_type": root.attrib.get("article-type"),
    }

    # ── Abstract ─────────────────────────────────────────────────────────────
    sections = []
    abstract_el = root.find("./front/article-meta/abstract")
    if abstract_el is not None:
        sections.append({
            "title":   "Abstract",
            "text":    _section_text(abstract_el),
            "sec_type": abstract_el.attrib.get("abstract-type", "abstract"),
        })

    # ── Body sections ────────────────────────────────────────────────────────
    for sec_el in root.findall("./body/sec"):
        title_el = sec_el.find("title")
        title    = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"
        text     = _section_text(sec_el)
        if len(text) < 100:          # skip boilerplate stubs
            continue
        sections.append({
            "title":    title,
            "text":     text,
            "sec_type": sec_el.attrib.get("sec-type"),   # populated when available
        })

    logger.info(f"PMID {pmid}: extracted {len(sections)} sections")
    return {"meta": meta, "sections": sections}


@task(name="process_article", retries=2, retry_delay_seconds=30)
def process_article(parsed: dict, sql_conn_str: str) -> list[dict]:
    """
    For each section: classify via LLM, extract metadata via LLM,
    assemble enriched chunk. Log failures to Azure SQL.
    Returns list of chunk dicts ready for upload.
    """
    logger  = get_run_logger()
    meta    = parsed["meta"]
    pmid    = meta["pmid"]
    chunks  = []

    for section in parsed["sections"]:
        title   = section["title"]
        text    = section["text"]
        preview = text[:300]

        # ── Step 1: classify section type ────────────────────────────────────
        # Use JATS sec-type attribute if available, otherwise ask the LLM
        if section.get("sec_type") and section["sec_type"] not in (None, ""):
            section_type = section["sec_type"].lower()
        else:
            result = _ollama_json(CLASSIFY_PROMPT.format(title=title, preview=preview))
            if result is None or "section_type" not in result:
                _log_failure(
                    conn_str=sql_conn_str,
                    pmid=pmid,
                    section_title=title,
                    failure_type="classify_failed",
                    raw_response=str(result),
                )
                logger.warning(f"PMID {pmid} | section '{title}' — classification failed, flagging")
                chunks.append(_flagged_chunk(meta, title, text))
                continue
            section_type = result["section_type"].lower()

        # Discard low-value sections
        if section_type in DISCARD_TYPES:
            logger.info(f"PMID {pmid} | section '{title}' — discarded ({section_type})")
            continue

        # ── Step 2: extract clinical metadata ────────────────────────────────
        meta_result = _ollama_json(EXTRACT_PROMPT.format(text=text[:2000]))
        if meta_result is None:
            _log_failure(
                conn_str=sql_conn_str,
                pmid=pmid,
                section_title=title,
                failure_type="extract_failed",
                raw_response="None",
            )
            logger.warning(f"PMID {pmid} | section '{title}' — extraction failed, flagging")
            chunks.append(_flagged_chunk(meta, title, text, section_type=section_type))
            continue

        # ── Step 3: assemble enriched chunk ──────────────────────────────────
        chunks.append({
            "pmid":         pmid,
            "year":         meta.get("year"),
            "journal":      meta.get("journal"),
            "article_type": meta.get("article_type"),
            "section_title": title,
            "section_type": section_type,
            "text":         text,
            "cancer_type":  meta_result.get("cancer_type", []),
            "treatments":   meta_result.get("treatments", []),
            "study_design": meta_result.get("study_design"),
            "phase":        meta_result.get("phase"),
            "sample_size":  meta_result.get("sample_size"),
            "prompt_version": PROMPT_VERSION,
            "flagged":      False,
        })

    logger.info(f"PMID {pmid}: assembled {len(chunks)} chunks")
    return chunks


@task(name="upload_processed", retries=3, retry_delay_seconds=10)
def upload_processed(pmid: str, chunks: list[dict], blob_creds: AzureBlobStorageCredentials) -> None:
    """Upload enriched chunk array as {pmid}.json to the processed container."""
    logger    = get_run_logger()
    conn_str  = blob_creds.connection_string.get_secret_value()
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client(BLOB_PROCESSED)
    if not container.exists():
        container.create_container()

    payload = json.dumps({"pmid": pmid, "chunks": chunks}, ensure_ascii=False)
    container.upload_blob(name=f"{pmid}.json", data=payload.encode("utf-8"), overwrite=True)
    logger.info(f"PMID {pmid}: uploaded {len(chunks)} chunks to {BLOB_PROCESSED}")


# ── SQL helpers ───────────────────────────────────────────────────────────────

def _log_failure(conn_str: str, pmid: str, section_title: str,
                 failure_type: str, raw_response: str) -> None:
    """Write a failure record to Azure SQL parse_failures table."""
    try:
        with _get_conn(conn_str) as conn:
            conn.execute(
                """
                INSERT INTO parse_failures
                    (pmid, section_title, failure_type, raw_response, prompt_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                pmid,
                section_title[:500],
                failure_type,
                raw_response[:4000],
                PROMPT_VERSION,
                datetime.now(timezone.utc),
            )
            conn.commit()
    except Exception:
        # Failure logging should never crash the pipeline
        pass


def _log_run(conn_str: str, run_id: str, started_at: datetime,
             articles_attempted: int, articles_succeeded: int,
             chunks_produced: int, failures: int) -> None:
    """Write a run summary record to Azure SQL parse_runs table."""
    try:
        with _get_conn(conn_str) as conn:
            conn.execute(
                """
                INSERT INTO parse_runs
                    (run_id, started_at, completed_at, articles_attempted,
                     articles_succeeded, chunks_produced, failures, prompt_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_id,
                started_at,
                datetime.now(timezone.utc),
                articles_attempted,
                articles_succeeded,
                chunks_produced,
                failures,
                PROMPT_VERSION,
            )
            conn.commit()
    except Exception:
        pass


def _flagged_chunk(meta: dict, title: str, text: str, section_type: str = "unclassified") -> dict:
    """Return a minimal chunk flagged for review when LLM calls fail."""
    return {
        "pmid":           meta["pmid"],
        "year":           meta.get("year"),
        "journal":        meta.get("journal"),
        "article_type":   meta.get("article_type"),
        "section_title":  title,
        "section_type":   section_type,
        "text":           text,
        "cancer_type":    [],
        "treatments":     [],
        "study_design":   None,
        "phase":          None,
        "sample_size":    None,
        "prompt_version": PROMPT_VERSION,
        "flagged":        True,
    }


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(name="parse_flow", log_prints=True)
def parse_flow():
    logger     = get_run_logger()
    started_at = datetime.now(timezone.utc)
    run_id     = runtime.flow_run.id

    blob_creds   = AzureBlobStorageCredentials.load("fiu-azure-blob-creds")
    sql_conn_str = Secret.load("azure-sql-connection-string").get()

    # List raw PMIDs available in Blob
    raw_conn_str  = blob_creds.connection_string.get_secret_value()
    raw_container = BlobServiceClient.from_connection_string(raw_conn_str).get_container_client(BLOB_RAW)
    all_pmids     = [b["name"].removesuffix(".xml") for b in raw_container.list_blobs()]

    # Idempotency — skip already processed
    processed = get_processed_pmids.submit(blob_creds).result()
    pmids     = [p for p in all_pmids if p not in processed]
    logger.info(f"{len(processed)} already processed, {len(pmids)} remaining")

    # Run-level counters
    articles_attempted  = len(pmids)
    articles_succeeded  = 0
    chunks_produced     = 0
    failures            = 0

    # Process MAX_WORKERS articles concurrently
    for i in range(0, len(pmids), MAX_WORKERS):
        window = pmids[i:i + MAX_WORKERS]

        # Download and parse JATS (fast, deterministic)
        download_futures = [download_raw.submit(pmid, blob_creds) for pmid in window]
        parse_futures    = [
            parse_jats.submit(pmid, f.result())
            for pmid, f in zip(window, download_futures)
        ]

        # LLM classification + extraction + chunk assembly
        process_futures = [
            process_article.submit(f.result(), sql_conn_str)
            for f in parse_futures
        ]

        # Upload enriched chunks and accumulate stats
        upload_futures = []
        for pmid, f in zip(window, process_futures):
            try:
                chunks = f.result()
                chunk_count = len(chunks)
                flagged     = sum(1 for c in chunks if c.get("flagged"))
                chunks_produced    += chunk_count
                failures           += flagged
                articles_succeeded += 1
                upload_futures.append(upload_processed.submit(pmid, chunks, blob_creds))
            except Exception as e:
                logger.error(f"PMID {pmid} failed entirely: {e}")
                failures += 1

        for uf in upload_futures:
            uf.result()

        logger.info(f"Completed window {i // MAX_WORKERS + 1} / {-(-len(pmids) // MAX_WORKERS)}")

    # Write run summary to Azure SQL
    _log_run(
        conn_str=sql_conn_str,
        run_id=str(run_id),
        started_at=started_at,
        articles_attempted=articles_attempted,
        articles_succeeded=articles_succeeded,
        chunks_produced=chunks_produced,
        failures=failures,
    )
    logger.info(
        f"Run complete — attempted: {articles_attempted}, "
        f"succeeded: {articles_succeeded}, "
        f"chunks: {chunks_produced}, "
        f"failures: {failures}"
    )


if __name__ == "__main__":
    parse_flow()