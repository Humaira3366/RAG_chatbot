# RAG_chatbot

📊 Data Flow Diagram (Level 1)
mermaid
Copy
Edit
graph TD
    A[User Uploads PDF] --> B[PDF Text Extraction (PyPDF)]
    B --> C[Text Chunking]
    C --> D[Embeddings Generation (SentenceTransformer)]
    D --> E[Store in Vector DB (FAISS)]
    F[User Query] --> G[Retriever]
    G --> E
    E --> H[Relevant Chunks]
    H --> I[LLM (Ollama - DeepSeek)]
    I --> J[Generated Answer]
    J --> K[Streamlit UI]

A Retrieval-Augmented Generation (RAG) architecture to build a chatbot that can answer user questions based on uploaded PDF content without needing OpenAI API keys. Here's a step-by-step explanation:

PDF Upload & Extraction

The user uploads a PDF via the Streamlit UI.

The text is extracted using PyPDF.

Text Chunking & Embedding

The extracted text is split into manageable chunks.

Each chunk is converted into a vector (embedding) using a SentenceTransformer model like all-MiniLM-L6-v2.

Vector Database (FAISS)

All vectorized chunks are stored in a FAISS vector store for fast similarity search.

User Query Processing

When a user enters a query, it is also embedded and matched against the FAISS store to retrieve the most relevant chunks.

Answer Generation (RAG)

The retrieved chunks and the user’s query are passed to a local LLM (DeepSeek model via Ollama) for answer generation.

Response Display

The generated answer is displayed back to the user on the Streamlit app interface.
