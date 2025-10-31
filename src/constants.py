from pathlib import Path
import yaml

DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 100_000
DEFAULT_GET_UNIQUE_DOCS = True  # Fixed typo: was DEFAUL_GET_UNIQUE_DOCS

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
NASA_MODEL = "nasa-impact/nasa-smd-ibm-v0.1"

MODEL_CONTEXT_SIZE = 128_000
TOKEN_OVERFLOW_LIMIT = 7_000

MCP_MAX_TOP_N = 20

EVE_PUBLIC_COLLECTION_NAME = "EVE open-access"

# Fallback LLM options
FALLBACK_LLM = "mistral-vanilla"  # Vanilla Mistral 3.2 24B as fallback

RERANKER_MODEL = "BAAI/bge-reranker-large"

WILEY_PUBLIC_COLLECTIONS = [
    {
        "name": "Wiley AI Gateway",
        "description": """This dataset contains 67,507 scholarly articles spanning a wide range of subject areas within the Earth and Environmental Sciences. The bulk of the content is drawn from Wiley's proprietary Earth Science, Meteorology, Climate Science, and Environmental Studies journals.

In addition to Wiley-owned publications, the dataset includes a significant body of work from the American Geophysical Union (AGU), representing key contributions in geophysics, atmospheric science, hydrology, and oceanography. Complementing these are notable clusters of society-owned titles, such as those published by the Royal Meteorological Society and the British Ecological Society, which extend coverage into meteorology, ecology, and environmental biology.

Together, this collection provides a comprehensive representation of the Earth and Environmental Sciences, blending proprietary Wiley titles with society publications to create a rich corpus of contemporary and legacy research outputs.""",
        "num_documents": 67507,
    },
]

PUBLIC_COLLECTIONS = [
    {
        "name": "ESA EO Knowledge Base",
        "description": "Curated collection of resources from ESA-related platforms and portals. It includes materials from ESA Earth Online, the Newcomers Earth Observation Guide, EO Portal, Sentiwiki, EO for Society publications, the CEOS ESA Catalogue, and the ESA Open Science Catalog. The dataset covers heterogeneous content such as web articles, technical documentation, instruments, datasets, and applications. Metadata has been systematically extracted and obtained, including URLs and titles.",
    },
    {
        "name": "esa-rag-scraped-qwen3",
        "description": "Curated collection of resources from ESA-related platforms and portals. It includes materials from ESA Earth Online, the Newcomers Earth Observation Guide, EO Portal, Sentiwiki, EO for Society publications, the CEOS ESA Catalogue, and the ESA Open Science Catalog. The dataset covers heterogeneous content such as web articles, technical documentation, instruments, datasets, and applications. Metadata has been systematically extracted and obtained, including URLs and titles. This collection contains around 100.000 documents.",
    },
    {
        "name": "qwen-512-filtered",
        "description": "Open-access collection of Earth Observation materials sourced from publishers and platforms such as MDPI, Springer, IOPscience, SagePub, EOGE, EOS, ISPRS,  and others. The dataset spans a wide range of content types, including research papers, journal articles, blog posts, and web pages. Alongside the documents, metadata has been systematically extracted to facilitate search and downstream analysis. All collected resources are compliant with current legislation regarding data use and accessibility. This collection contains about 250.00 documents.",
    },
    {
        "name": "wikipedia_eo_dump",
        "description": "This collection brings together Wikipedia articles related to Earth Observation (EO). The content is intended to provide accessible, introductory information about EO concepts, technologies, and organizations active in the field. Please note that these articles are not peer-reviewed scientific publications. Instead, they are written for a general audience and aim to give broad overviews rather than in-depth, expert analyses. Users should treat this collection as a starting point for understanding EO, and complement it with specialized, peer-reviewed sources when deeper or technical knowledge is required. This collection contains about 2000 documents.",
    },
]

STAGING_PUBLIC_COLLECTIONS = [
    {
        "name": "esa-data-qwen-1024",
        "description": "ESA data with Qwen-1024 for testing",
    },
    {
        "name": "Wikipedia EO",
        "description": "This collection brings together Wikipedia articles related to Earth Observation (EO). The content is intended to provide accessible, introductory information about EO concepts, technologies, and organizations active in the field. Please note that these articles are not peer-reviewed scientific publications. Instead, they are written for a general audience and aim to give broad overviews rather than in-depth, expert analyses. Users should treat this collection as a starting point for understanding EO, and complement it with specialized, peer-reviewed sources when deeper or technical knowledge is required. This collection contains about 2000 documents.",
    },
    {"name": "qwen-512-filtered", "description": "ESA data with Qwen-512 for testing"},
    {
        "name": "satcom-chunks-collection",
        "alias": "SATCOM Technical Knowledge Base",
        "description": "Curated collection of resources on Satellite Communications (SATCOM) sourced from peer-reviewed publishers and journals, including MDPI, Oxford University Press, Springer, IEEE, and other leading scientific platforms. The dataset covers a broad range of technical content such as research papers, review articles, standards, and technical documentation focused on communication systems, satellite payloads, link design, modulation, and emerging SATCOM technologies",
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
