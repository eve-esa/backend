from pathlib import Path
import yaml

DEFAULT_QUERY = "What is ESA?"
DEFAULT_COLLECTION = "esa-nasa-workshop"
DEFAULT_LLM = "eve-instruct-v0.1"  # or openai
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_K = 3
DEFAULT_SCORE_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 1500
DEFAULT_GET_UNIQUE_DOCS = True  # Fixed typo: was DEFAUL_GET_UNIQUE_DOCS

DEFAULT_EMBEDDING_MODEL = "nasa-impact/nasa-smd-ibm-st-v2"
NASA_MODEL = "nasa-impact/nasa-smd-ibm-v0.1"

MODEL_CONTEXT_SIZE = 128 * 1024

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
        "name": "eve-esa-data",
        "description": "Open-access collection of Earth Observation materials sourced from publishers and platforms such as MDPI, Springer, IOPscience, SagePub, EOGE, EOS, ISPRS, EUMETSAT, and others. The dataset spans a wide range of content types, including research papers, journal articles, blog posts, and web pages. Alongside the documents, metadata has been systematically extracted to facilitate search and downstream analysis. All collected resources are compliant with current legislation regarding data use and accessibility.",
    },
    {
        "name": "ESA EO Knowledge Base",
        "description": "Curated collection of resources from ESA-related platforms and portals. It includes materials from ESA Earth Online, the Newcomers Earth Observation Guide, EO Portal, Sentiwiki, EO for Society publications, the CEOS ESA Catalogue, and the ESA Open Science Catalog. The dataset covers heterogeneous content such as web articles, technical documentation, instruments, datasets, and applications. Metadata has been systematically extracted and obtained, including URLs and titles.",
    },
]


_BASE_DIR = Path(__file__).parent
_POLICY_PATH = _BASE_DIR / "templates" / "donotknow.yaml"
with open(_POLICY_PATH, "r") as _f:
    POLICY_PROMPT = yaml.safe_load(_f)["policy_prompt"]

POLICY_NOT_ANSWER = """Oops, that's outside my orbit! I'm here to talk about Earth Observation. If you'd like, we can explore satellites, remote sensing, or climate topics instead."""
