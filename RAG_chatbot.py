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

# Sidebar: File upload remains the same
st.sidebar.title("🤖 I’m your PDF Buddy! Ask me anything from your file!")
st.sidebar.title("📂 PDF Chatbot")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])
st.sidebar.markdown("Limit: 200MB per file • Format: PDF")

# Sidebar Toggle Section
option = st.sidebar.radio("🔍 View Sidebar Info:", ["PDF Summary", "Tips", "Recent Questions"])

if option == "PDF Summary":
    st.sidebar.markdown("### 📄 Summary")
    st.sidebar.markdown("""
    - **Pages**: 12  
    - **Main Topic**: Big Transformer  
    - **Focus**: AI-based Machine Translation  
    - **Highlights**: Label Smoothing, BLEU Scores  
    """)

elif option == "Tips":
    st.sidebar.markdown("### 💡 How to Use")
    st.sidebar.markdown("""
    1. Upload a PDF file  
    2. Ask specific questions like:  
       - "What is the paper about?"  
       - "Explain evaluation results"  
    3. Get answers with citation support  
    """)

elif option == "Recent Questions":
    st.sidebar.markdown("### 📝 Recent Questions")
    # You can store these in session_state when users ask a question
    recent_questions = st.session_state.get("recent_questions", [
        "What is the core of this PDF?",
        "List evaluation results",
        "Explain label smoothing"
    ])
    for q in recent_questions:
        st.sidebar.markdown(f"- {q}")

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

