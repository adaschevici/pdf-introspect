import streamlit as st
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


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
    return chunks

def get_vector_store(text_chunks):
    vector_store = {}
    for idx, chunk in enumerate(text_chunks):
        vector_store[idx] = chunk
    return vector_store


def main():
    settings = Settings()
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
