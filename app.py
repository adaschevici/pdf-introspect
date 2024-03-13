import streamlit as st

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")
    st.header("Chat with multiple PDFs :books:")

    st.text_input("Ask a question about the PDFs")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload some PDFs and click process", type="pdf", accept_multiple_files=True)

if __name__ == '__main__':
    main()
