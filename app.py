import streamlit as st
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from PyPDF2 import PdfReader


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
                raw_text = get_text_from_pdf(pdf_docs)
                st.write(raw_text)


if __name__ == "__main__":
    main()
