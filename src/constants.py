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

# Fallback LLM options
FALLBACK_LLM = "mistral-vanilla"  # Vanilla Mistral 3.2 24B as fallback

RERANKER_MODEL = "BAAI/bge-reranker-large"
