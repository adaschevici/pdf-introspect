import time
import streamlit as st
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from html_template import css, bot_template, user_template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder


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
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="avsolatorio/GIST-Embedding-v0", model_kwargs={"device": "mps"})
    vector_store = Qdrant.from_documents(
        text_chunks,
        embeddings,
        url=qdrant_url,
        collection_name="pdfs",
        force_recreate=True,
    )
    return vector_store

def get_context_retriever_chain(vector_store, settings):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=settings.openai_api_key.get_secret_value(),
    )

    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        # HumanMessagePromptTemplate.from_template("{question}"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])
    # prompt = prompt.format_messages(question=question)
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversation_chain(retriever_chain, settings):
    system_template = """
        Question: Please answer the question with citation to the paragraphs.
        For every sentence you write, cite the book name and paragraph number as <id_x_x> 
 
         At the end of your commentary: 
             1. Add key words from the book paragraphs.  
             2. Suggest a further question that can be answered by the paragraphs provided.  
             3. Create a sources list of book names, paragraph Number author name, and a link for each book you cited.
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=settings.openai_api_key.get_secret_value(),
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("system", "Answer the users's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        # HumanMessagePromptTemplate.from_template("{question}"),
        ("user", "{input}"),
    ])
    # prompt = prompt.format_messages(question=question)
    conversation_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, conversation_chain)

def handle_user_input(user_question, settings):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, settings=settings)
    conversation_rag_chain = get_conversation_chain(retriever_chain, settings=settings)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_question
    })
    return response.get("answer")


def main():
    import os
    settings = Settings()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.hf_access_token.get_secret_value()
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
    st.set_page_config(
        page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide"
    )

    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! How can I help you today?")
        ]

    st.header("Chat with multiple PDFs :books:")


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload some PDFs and click process", type="pdf", accept_multiple_files=True
        )


    # create conversation chain
    if len(pdf_docs) == 0:
        st.info("Please upload some PDFs to start chatting.")
    else:
        with st.sidebar:
            if st.button("Process"):
                with st.spinner("Processing..."):
                    # get raw content from pdf
                    raw_text = get_text_from_pdf(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)

                    if "vector_store" not in st.session_state:
                        start = time.time()
                        st.session_state.vector_store = get_vector_store(text_chunks)
                        end = time.time()
                        # create vector store for each chunk
                        st.write(f"Time taken to create vector store: {end - start}")
        if "vector_store" not in st.session_state:
            st.info("Please process the PDFs first.")
        else:
            user_question = st.text_input("Ask a question about the PDFs...")
            if user_question:
            # st.write(user_template.format(message=user_question), unsafe_allow_html=True)
                response = handle_user_input(user_question, settings=settings)
                st.write(bot_template.format(message=response), unsafe_allow_html=True)
                #            if "vector_store" in st.session_state:
                #                st.session_state.conversation = get_conversation_chain(st.session_state.vector_store, user_question, settings=settings)


if __name__ == "__main__":
    main()
