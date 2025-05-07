import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import os

# Set page config
st.set_page_config(page_title="RAG-Powered PDF Chatbot", layout="wide")

st.title("📄 RAG-Powered Web PDF Chatbot (Ollama + DeepSeek)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Store PDF in temp
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    # Load & split PDF
    loader = PyPDFLoader(tmp_pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Embed & store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()

    # Load Ollama LLM (DeepSeek)
    llm = Ollama(model="deepseek-coder")

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User query
    user_question = st.text_input("Ask a question from the PDF")

    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_question)
            st.success(response)

    # Clean up
    os.remove(tmp_pdf_path)
else:
    st.info("👆 Upload a PDF to get started.")
