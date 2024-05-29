import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader


load_dotenv()
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")


def read_urls_from_file(file_path):
    """
    Reads a text file where each line contains a URL and returns a list of all URLs.

    :param file_path: Path to the text file containing URLs
    :return: List of URLs
    """
    urls = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                url = line.strip()  # Remove any surrounding whitespace
                if url:  # Only add non-empty lines
                    urls.append(url)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return urls


if __name__ == "__main__":
    url_list = read_urls_from_file("rag_data.txt")
    all_docs = []

    for url in url_list:
        loader = WebBaseLoader(url)

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed", mistral_api_key=mistral_api_key
    )

    qdrant = Qdrant.from_documents(
        all_docs,
        embeddings,
        url=qdrant_url,
        prefer_grpc=True,
        api_key=qdrant_api_key,
        collection_name="test_llm4eo",
        force_recreate=True,
    )
