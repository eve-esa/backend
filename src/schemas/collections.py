from pydantic import BaseModel

class CollectionRequest(BaseModel):
    embeddings_model: str =  "nasa-impact/nasa-smd-ibm-st-v2"