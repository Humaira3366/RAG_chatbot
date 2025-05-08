import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import os

# Set app page config
st.set_page_config(page_title="RAG-Powered PDF Chatbot", layout="wide", page_icon="📚")

# Sidebar
st.sidebar.image("https://img.icons8.com/external-flat-juicy-fish/60/null/external-chatbot-customer-support-flat-flat-juicy-fish.png", width=60)
st.sidebar.title("📂 PDF Chatbot")
st.sidebar.write("Upload your PDF and ask questions directly from its content.")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📄 Upload your PDF", type="pdf")

# Main title and description
st.markdown("## 🤖 RAG-Powered Web PDF Chatbot (Ollama + DeepSeek)")
st.markdown("> Ask contextual questions from any uploaded PDF document using **Retrieval-Augmented Generation** with **LLM** support.")
st.markdown("---")

# Store PDF temporarily
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

    # Load LLM
    llm = Ollama(model="deepseek-coder")

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Input for question
    st.markdown("### ❓ Ask a Question from your PDF")
    user_question = st.text_input("Type your question here:")

    if user_question:
        with st.spinner("🔍 Thinking..."):
            response = qa_chain.run(user_question)
            st.success(response)

    # Cleanup
    os.remove(tmp_pdf_path)
else:
    st.info("⬅️ Upload a PDF from the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color: gray;'>Built with ❤️ using Streamlit, LangChain & Ollama</p>",
    unsafe_allow_html=True
)
