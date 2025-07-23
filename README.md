# 🤖 RAG Chatbot – PDF-Powered Question Answering App

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **FAISS**, **SentenceTransformers**, and a **local LLM (Ollama)**. It allows users to upload a PDF and ask questions from its content – no OpenAI key needed!

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/8d3dc36e-b3fb-4f15-8237-70342fe5d35e" />


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
### 🧠 Architecture Overview

Upload PDF → Extract → Chunk → Embed → FAISS Store
↑ ↓
User Query → Embed → Retrieve Chunks
↓
Local LLM → Generate Answer → Streamlit UI


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


