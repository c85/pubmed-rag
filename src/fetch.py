import time
import requests
import xml.etree.ElementTree as ET
from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from azure.storage.blob import BlobServiceClient
from prefect_azure import AzureBlobStorageCredentials

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ELINK_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_QUERY = (
    "Neoplasms[MeSH] AND "
    "(Child[MeSH] OR Adolescent[MeSH] OR Infant[MeSH]) AND "
    '"free full text"[sb] AND '
    '("2015/01/01"[dp] : "3000/12/31"[dp]) AND '
    "(Review[pt] OR Clinical Trial[pt])"
)
BATCH_SIZE   = 25
MAX_WORKERS  = 3
BLOB_CONTAINER = "pubmed-raw"

@task(name="get_pmids", retries=3, retry_delay_seconds=10)
def get_pmids(api_key: str) -> list[str]:
    """Run esearch and return the full list of PMIDs matching the corpus query."""
    logger = get_run_logger()

    # First get total count
    params = {
        "db": "pubmed",
        "term": PUBMED_QUERY,
        "retmode": "json",
        "rettype": "count",
        "api_key": api_key,
    }
    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    total = int(resp.json()["esearchresult"]["count"])
    logger.info(f"Total matching articles: {total}")

    # Fetch all PMIDs in one call using retmax
    params.update({"rettype": "uilist", "retmax": total})
    resp = requests.get(ESEARCH_URL, params=params, timeout=60)
    resp.raise_for_status()
    pmids = resp.json()["esearchresult"]["idlist"]
    logger.info(f"Retrieved {len(pmids)} PMIDs")
    return pmids

@task(name="batch_pmids")
def batch_pmids(pmids: list[str], batch_size: int = BATCH_SIZE) -> list[list[str]]:
    """Split PMID list into batches for concurrent efetch calls."""
    batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
    get_run_logger().info(f"Split into {len(batches)} batches of up to {batch_size}")
    return batches

@task(name="fetch_batch", retries=3, retry_delay_seconds=15)
def fetch_batch(batch: list[str], api_key: str) -> dict[str, str]:
    """
    Convert a batch of PubMed IDs to PMC IDs via elink, fetch full-text XML,
    and return a dict of {pmid: xml_string} for each successfully parsed article.
    """
    logger = get_run_logger()

    # Step 1: Convert PMIDs → PMCIDs
    elink_resp = requests.get(ELINK_URL, params={
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": ",".join(batch),
        "retmode": "xml",
        "api_key": api_key,
    }, timeout=30)
    elink_resp.raise_for_status()

    elink_root = ET.fromstring(elink_resp.text)
    pmcids = [
        el.text
        for el in elink_root.findall(".//LinkSetDb[LinkName='pubmed_pmc']/Link/Id")
        if el.text
    ]

    if not pmcids:
        logger.warning(f"No PMC IDs found for batch starting {batch[0]}")
        return {}

    logger.info(f"Resolved {len(pmcids)} PMC IDs from {len(batch)} PMIDs")

    # Step 2: Fetch full-text XML one article at a time so a single large
    # article can't cause a ChunkedEncodingError that drops the whole batch.
    articles = {}
    for pmcid in pmcids:
        try:
            resp = requests.get(EFETCH_URL, params={
                "db": "pmc",
                "id": pmcid,
                "retmode": "xml",
                "rettype": "full",
                "api_key": api_key,
            }, timeout=120, stream=True)
            resp.raise_for_status()
            content = b"".join(resp.iter_content(chunk_size=65536))
            root = ET.fromstring(content)
            for article_el in root.findall("article"):
                pmid_el = article_el.find(
                    "./front/article-meta/article-id[@pub-id-type='pmid']"
                )
                if pmid_el is not None and pmid_el.text:
                    articles[pmid_el.text.strip()] = ET.tostring(article_el, encoding="unicode")
        except Exception as e:
            logger.warning(f"Skipping PMC ID {pmcid}: {e}")
        time.sleep(0.11 * MAX_WORKERS)  # ~3 workers × ~3 req/s each = ~9 req/s total

    logger.info(f"Parsed {len(articles)} articles from {len(pmcids)} PMC IDs")
    return articles

@task(name="get_uploaded_pmids")
def get_uploaded_pmids(blob_storage_credentials: AzureBlobStorageCredentials) -> set[str]:
    """Return the set of PMIDs already present in the blob container."""
    assert blob_storage_credentials.connection_string is not None
    conn_str = blob_storage_credentials.connection_string.get_secret_value()
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client(BLOB_CONTAINER)
    if not container.exists():
        return set()
    return {b["name"].removesuffix(".xml") for b in container.list_blobs()}


@task(name="upload_raw", retries=3, retry_delay_seconds=10)
def upload_raw(articles: dict[str, str], blob_storage_credentials: AzureBlobStorageCredentials) -> None:
    logger = get_run_logger()
    if not articles:
        return
    assert blob_storage_credentials.connection_string is not None
    conn_str = blob_storage_credentials.connection_string.get_secret_value()
    container = BlobServiceClient.from_connection_string(conn_str).get_container_client(BLOB_CONTAINER)
    if not container.exists():
        container.create_container()
    uploaded = 0
    for pmid, xml in articles.items():
        container.upload_blob(name=f"{pmid}.xml", data=xml.encode("utf-8"), overwrite=True)
        uploaded += 1

    logger.info(f"Uploaded {uploaded} raw XML files to {BLOB_CONTAINER}")


@flow(name="fetch_flow", log_prints=True)
def fetch_flow():
    logger = get_run_logger()

    api_key = Secret.load("ncbi-api-key").get()
    blob_storage_credentials: AzureBlobStorageCredentials = AzureBlobStorageCredentials.load("fiu-azure-blob-creds")

    pmids = get_pmids.submit(api_key).result()
    uploaded = get_uploaded_pmids.submit(blob_storage_credentials).result()
    pmids = [p for p in pmids if p not in uploaded]
    logger.info(f"{len(uploaded)} already uploaded, {len(pmids)} remaining")
    batches = batch_pmids.submit(pmids).result()

    logger.info(f"Fetching {len(batches)} batches, {MAX_WORKERS} at a time")

    for i in range(0, len(batches), MAX_WORKERS):
        fetch_futures = [fetch_batch.submit(batch, api_key) for batch in batches[i:i + MAX_WORKERS]]
        upload_futures = []
        for future in fetch_futures:
            articles = future.result()
            upload_futures.append(upload_raw.submit(articles, blob_storage_credentials))
        for uf in upload_futures:
            uf.result()
        time.sleep(1)


if __name__ == "__main__":
    fetch_flow()