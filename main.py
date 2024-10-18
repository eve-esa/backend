from src.services.vector_store_manager import VectorStoreManager
from langchain_core.documents import Document
import openai
from openai import OpenAI
from rich.console import Console
from rich.text import Text
from pyfiglet import Figlet

from src.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY

if __name__ == "__main__":

    # "text-embedding-3-small"
    # "sentence-transformers/paraphrase-TinyBERT-L6-v2"
    # "mistral-embed"

    vector_store = VectorStoreManager(
        QDRANT_URL, QDRANT_API_KEY, embeddings_model="text-embedding-3-small"
    )

    query = "What is the European Space Agency?"

    results = vector_store.retrieve_documents_from_query(
        query=query,
        embeddings_model="mistral-embed",
        collection_name="test_llm4eo",
        score_threshold=0.6,
        k=1,
        get_unique_docs=True,
    )

    retrieved_documents = [result.payload["page_content"] for result in results]

    context = "\n".join(retrieved_documents)
    print("CONTEXT: ", context)

    res = vector_store.generate_answer(query=query, context=context, llm="llama-3.1")
    console = Console()

    # Generate big text using pyfiglet
    figlet = Figlet(font="slant")
    big_text = figlet.renderText("Eve")

    # Style the big text with a color using rich Text
    styled_text = Text(big_text, style="bold magenta")

    # Print the styled text
    console.print(styled_text)
    print(res)
