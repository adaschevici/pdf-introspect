import streamlit as st
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="config.env", env_file_encoding="utf-8")
    hf_access_token: SecretStr = Field(alias="HF_ACCESS_TOKEN")
    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")

settings = Settings()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about the PDFs")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload some PDFs and click process", type="pdf", accept_multiple_files=True)
        st.button("Process")

if __name__ == '__main__':
    print(settings)
    # main()
