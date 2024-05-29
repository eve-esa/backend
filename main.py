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
from qdrant_client import QdrantClient
import streamlit as st

load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
mistral_api_key = os.environ.get("MISTRAL_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=mistral_api_key)


client = QdrantClient(
    qdrant_url,
    api_key=qdrant_api_key,  # For Qdrant Cloud, None for local instance
)

doc_store = Qdrant(
    client=client,
    collection_name="test_llm4eo",
    embeddings=embeddings,
)


retriever = doc_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
client = QdrantClient(url="http://localhost:6333")


model = ChatMistralAI(mistral_api_key=mistral_api_key)
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)


st.title("Pi School - EOVE")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Create a retrieval chain to answer questions
        document_chain = create_stuff_documents_chain(model, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": query})
        st.write(response["answer"])
        for doc in response["context"]:
            st.text(doc.metadata["title"])
            st.markdown(doc.metadata["source"])
    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
