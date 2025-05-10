# 🤖 RAG Chatbot – PDF-Powered Question Answering App

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **FAISS**, **SentenceTransformers**, and a **local LLM (Ollama)**. It allows users to upload a PDF and ask questions from its content – no OpenAI key needed!

---

## 🧠 How It Works (Simplified Flow)

📄 **User Uploads PDF**  
🔍 **Text is extracted using PyPDF**  
🧩 **Text is chunked**  
🧠 **Chunks embedded via SentenceTransformer (`all-MiniLM-L6-v2`)**  
🗃️ **FAISS stores chunk vectors for similarity search**  
💬 **User query is embedded and matched**  
🧠 **Relevant chunks + query sent to Ollama (DeepSeek LLM)**  
🧾 **Answer generated and shown via Streamlit**

---

## 📸 Architecture Overview

```mermaid
graph TD
    A[📤 User Uploads PDF] --> B[🔍 Extract Text (PyPDF)]
    B --> C[🧩 Chunk Text]
    C --> D[🧠 Generate Embeddings (SentenceTransformer)]
    D --> E[📦 Store Vectors in FAISS]
    E --> F[💬 User Query]
    F --> G[🔍 Match Relevant Chunks (FAISS)]
    G --> H[🤖 Answer using Ollama LLM (DeepSeek)]
    H --> I[🖥️ Streamlit Display]

🚀 Features
✅ Upload PDFs and extract text

✅ Chunk content dynamically

✅ Embed using pretrained SentenceTransformer

✅ Store and retrieve chunks via FAISS

✅ Generate accurate answers using local LLM

✅ Simple Streamlit UI (fast + local)

🛠️ Tech Stack
| Tool                   | Role                 |
| ---------------------- | -------------------- |
| `Streamlit`            | UI & interactions    |
| `PyPDF`                | PDF text extraction  |
| `SentenceTransformers` | Vector embeddings    |
| `FAISS`                | Similarity search    |
| `Ollama` + `DeepSeek`  | Local language model |
| `Python-dotenv`        | Env config           |

📦 Installation

git clone https://github.com/Humaira3366/RAG_chatbot.git
cd RAG_chatbot
pip install -r requirements.txt

Ensure you have Ollama installed and the deepseek-coder model available:
ollama run deepseek-coder

▶️ Run the App
streamlit run app.py
📝 Usage Example
Upload any PDF with readable text.

Ask a question like:
💬 "What is the objective of this paper?"

Wait for the generated answer from the LLM.

See results below the question box.

📌 Coming Soon
🗃️ Chat history memory

🧾 PDF summary generator

🌐 Multi-PDF RAG support

🙋‍♀️ Author
Name: Humaira Fathima N
LinkedIn: www.linkedin.com/in/humairafathima-n-778415295
Email: humaira2004super@gmail.com

⭐ Star this repo if you found it useful!

---

Would you like a matching `requirements.txt` or a custom badge (e.g., PDF-powered 🧠 bot)?

