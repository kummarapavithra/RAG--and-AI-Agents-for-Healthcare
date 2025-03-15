# Document Chatbot with RAG & Llama 

## Infosys Springboard Internship Project  
AI-powered document interaction system using Retrieval-Augmented Generation (RAG) and large language models.

---

### 📌 Project Overview
A Streamlit-based chatbot that allows users to:
1. Upload PDF/text documents
2. Ask questions about document content
3. Receive AI-generated answers with context from uploaded files

**Core Technologies**:
- **RAG Architecture** for context-aware responses
- **Llama 3-8B** via Groq API for response generation
- **FAISS** for semantic search and vector storage

---

### 🚀 Key Features
✅ Document processing (PDF/TXT)  
✅ Contextual question answering  
✅ Real-time response streaming  
✅ Persistent chat history  
✅ Local vector storage  

---

### 🔧 Tech Stack
| Component       | Technology           |
|-----------------|----------------------|
| Frontend        | Streamlit            |
| LLM             | Llama 3-8B (Groq)    |
| Embeddings      | HuggingFace MiniLM   |
| Vector DB       | FAISS                |
| File Processing | PyPDF2 + Python std  |

---

### 🛠️ Setup & Installation
1. **Clone repository**

2. **Set up environment**
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

3. **Configure API key**
Create .env file
GROQ_API_KEY=your_groq_api_key_here

4. **Run application**
streamlit run app.py

📚 Usage Guide
1.Upload documents via sidebar
2.Wait for processing confirmation
3.Ask questions in chat interface
4.View real-time AI responses


 

**Learning Outcomes**
Through this internship project, I gained experience in:
1.Implementing RAG architecture
2.Working with large language models
3.Building vector databases
4.Developing streaming web applications
5.Managing API integrations


