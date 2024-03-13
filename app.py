import streamlit as st
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from qdrant_client import models


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="config.env", env_file_encoding="utf-8")
    hf_access_token: SecretStr = Field(alias="HUGGINGFACEHUB_API_TOKEN")
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")


def get_text_from_pdf(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return [Document(page_content=chunk) for chunk in chunks if chunk.strip()]


def get_vector_store(text_chunks, qdrant_url="http://localhost:6333"):
    embeddings = OpenAIEmbeddings()
    vector_store = Qdrant.from_documents(
        text_chunks,
        embeddings,
        url=qdrant_url,
        collection_name="pdfs",
        force_recreate=True,
    )
    return vector_store


def main():
    import os
    settings = Settings()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.hf_access_token.get_secret_value()
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    st.set_page_config(
        page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide"
    )
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about the PDFs")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload some PDFs and click process", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):
                # get raw content from pdf
                raw_text = get_text_from_pdf(pdf_docs)

                # split the text into chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store for each chunk
                vector_store = get_vector_store(text_chunks)


if __name__ == "__main__":
    main()
