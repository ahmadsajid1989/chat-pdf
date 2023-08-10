import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_texts):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text=raw_texts)


def get_vectors(text_chunks):
    model_name = "hkunlp/instructor-xl"
    model_kwargs = {'device': 'cpu'}
    hf = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,

    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=hf)
    return vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")
    st.header("Chat PDF :robot_face:")

    st.text_input("Ask a question about your document")

    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing your files.."):
                # get the text from pdf files
                raw_texts = get_pdf_texts(pdf_docs)
                # get texts in chunks
                text_chunks = get_text_chunks(raw_texts)
                # create vector store
                vector_store = get_vectors(text_chunks)


if __name__ == '__main__':
    main()
