from pathlib import Path

import yaml

DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 10
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_MAX_NEW_TOKENS = 100_000
DEFAULT_GET_UNIQUE_DOCS = True  # Fixed typo: was DEFAUL_GET_UNIQUE_DOCS

JSC_DEFAULT_EMBEDDING_MODEL = "alias-qwen3-4b-embeddings"
DEEPINFRA_DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBEDDING_MODEL = JSC_DEFAULT_EMBEDDING_MODEL
NASA_MODEL = "nasa-impact/nasa-smd-ibm-v0.1"

MODEL_CONTEXT_SIZE = 128_000
TOKEN_OVERFLOW_LIMIT = 7_000

MCP_MAX_TOP_N = 20
MAX_PRIVATE_DOCUMENTS = 150

# Documents attached to one conversation (POST /conversations/{id}/documents).
# They also count against MAX_PRIVATE_DOCUMENTS.
MAX_CONVERSATION_DOCUMENTS = 10
MAX_CONVERSATION_DOCUMENT_BYTES = 10 * 1024 * 1024
# Per file, after splitting: past it the embedding cannot finish within the
# deadline below, so the file is refused before any embedding call.
MAX_CONVERSATION_DOCUMENT_CHUNKS = 600
CONVERSATION_DOCUMENT_EXTENSIONS = (".pdf", ".txt", ".md")
# Whole request, parsing and embedding included. Below the 120 s CloudFront
# origin read timeout, with room for the rollback that runs after it expires.
CONVERSATION_DOCUMENT_DEADLINE_SECONDS = 90

EVE_PUBLIC_COLLECTION_NAME_PROD = "qwen-512-filtered"

PRIVATE_COLLECTION_NAME_PROD = "private-collections"
PRIVATE_COLLECTION_NAME_STAGING = "private-collections-staging"
PRIVATE_COLLECTION_NAME_DEV = "private-collections-dev"
ALL_PRIVATE_COLLECTION_NAMES = frozenset(
    {
        PRIVATE_COLLECTION_NAME_PROD,
        PRIVATE_COLLECTION_NAME_STAGING,
        PRIVATE_COLLECTION_NAME_DEV,
    }
)

PUBLIC_ENV_PROD = "prod"
PUBLIC_ENV_STAGING = "staging"


def private_collection_name_for(environment: str) -> str:
    """Qdrant collection that holds private user vectors for ``environment``.

    Prod keeps the historical name. Staging and dev use suffixes so all three
    backends can share one cluster without mixing tenants. Unknown values
    (including the legacy ``non-prod`` fallback) map to the dev collection.
    """
    if environment == "prod":
        return PRIVATE_COLLECTION_NAME_PROD
    if environment == "staging":
        return PRIVATE_COLLECTION_NAME_STAGING
    return PRIVATE_COLLECTION_NAME_DEV


# Fallback LLM options
FALLBACK_LLM = "mistral-vanilla"  # Vanilla Mistral 3.2 24B as fallback

WILEY_PUBLIC_COLLECTIONS = [
    {
        "name": "Wiley AI Gateway",
        "description": (
            "Peer-reviewed full text from Wiley's portfolio of more than 2,000 journals, "
            "across the sciences, medicine, engineering, business, law and the humanities. "
            "Coverage follows Wiley's publishing portfolio, so disciplines are not equally "
            "represented. Search covers article text only, not figures, images or "
            "supplementary data. Retracted articles are excluded."
        ),
    },
]

PUBLIC_COLLECTIONS = [
    {
        "name": "esa-rag-scraped-qwen3-newpipeline",
        "alias": "ESA EO Knowledge Base",
        "description": "Curated collection of resources from ESA-related platforms and portals. It includes materials from ESA Earth Online, the Newcomers Earth Observation Guide, EO Portal, Sentiwiki, EO for Society publications, the CEOS ESA Catalogue, and the ESA Open Science Catalog. The dataset covers heterogeneous content such as web articles, technical documentation, instruments, datasets, and applications. Metadata has been systematically extracted and obtained, including URLs and titles. This collection contains around 100.000 documents.",
    },
    {
        "name": "qwen-512-filtered",
        "alias": "EVE open access",
        "description": "Open-access collection of Earth Observation materials sourced from publishers and platforms such as MDPI, Springer, IOPscience, SagePub, EOGE, EOS, ISPRS,  and others. The dataset spans a wide range of content types, including research papers, journal articles, blog posts, and web pages. Alongside the documents, metadata has been systematically extracted to facilitate search and downstream analysis. All collected resources are compliant with current legislation regarding data use and accessibility. This collection contains about 250.00 documents.",
    },
    {
        "name": "wikipedia-512",
        "alias": "Wikipedia EO",
        "description": "This collection brings together Wikipedia articles related to Earth Observation (EO). The content is intended to provide accessible, introductory information about EO concepts, technologies, and organizations active in the field. Please note that these articles are not peer-reviewed scientific publications. Instead, they are written for a general audience and aim to give broad overviews rather than in-depth, expert analyses. Users should treat this collection as a starting point for understanding EO, and complement it with specialized, peer-reviewed sources when deeper or technical knowledge is required. This collection contains about 2000 documents.",
    },
]

SCRAPING_DOG_ALL_URLS = [
    "https://earth.esa.int/eogateway",
    "https://earthdata.nasa.gov",
    "https://dataspace.copernicus.eu/",
    "https://earthexplorer.usgs.gov",
    "https://earthobservations.org",
    "https://earthengine.google.com",
    "https://www.class.noaa.gov",
    "https://www.star.nesdis.noaa.gov",
    "https://ceos.org",
    "https://radiant.earth",
    "https://eos.com",
    "https://www.sentinel-hub.com/",
    "https://www.eumetsat.int/",
    "https://www.ecmwf.int/",
    "https://atmosphere.copernicus.eu/",
    "https://marine.copernicus.eu/",
    "https://land.copernicus.eu/en",
    "https://climate.copernicus.eu/",
    "https://emergency.copernicus.eu/",
    "https://nsidc.org/home",
    "https://climate.esa.int/en/",
    "https://cpom.org.uk/",
    "https://earth.esa.int/eogateway/search?category=campaigns",
    "https://philab.esa.int/",
]

_BASE_DIR = Path(__file__).parent
_POLICY_PATH = _BASE_DIR / "templates" / "donotknow.yaml"
with open(_POLICY_PATH, "r") as _f:
    POLICY_PROMPT = yaml.safe_load(_f)["policy_prompt"]

POLICY_NOT_ANSWER = """  Oops, that's outside my orbit! I'm here to talk about Earth Observation. If you'd like, we can explore satellites, remote sensing, or climate topics instead."""
