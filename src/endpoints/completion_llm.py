"""
Consider deleting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.config import Config
import runpod

config = Config()
router = APIRouter()


class CompletionRequest(BaseModel):
    query: str = "ESA is a space agency and"


@router.post("/completion_llm")
async def create_collection(request: CompletionRequest):
    endpoint = runpod.Endpoint(config.get_completion_llm_id())
    input_data = {"input": {"prompt": request.query}}

    try:
        print(f"Processing request for query: {request.query}")
        res = endpoint.run_sync(request_input=input_data, timeout=config.get_timeout())

        if not res or "choices" not in res[0] or not res[0]["choices"]:
            print(f"Invalid response structure: {res}")
            raise HTTPException(
                status_code=502, detail="Invalid response from LLM service"
            )

        print(f"Successfully processed request for query: {request.query}")
        return {"query": request.query, "response": res[0]["choices"][0]["tokens"][0]}

    except TimeoutError:
        print(f"Timeout occurred while processing the request : {str(e)}")
        raise HTTPException(status_code=504, detail="Job timed out")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
