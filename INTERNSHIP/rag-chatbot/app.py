import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="Document Chatbot")
st.title("📄Doc-MATE AI")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# File upload and processing
def process_files(files):
    text = ""
    for file in files:
        if file.type == "application/pdf":
            pdf = PdfReader(file)
            text += "\n".join(page.extract_text() for page in pdf.pages)
        elif file.type == "text/plain":
            text += file.read().decode("utf-8")
    return text

# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# Embedding and vector store creation
def create_vector_store(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embeddings)

# RAG pipeline
def retrieve_and_generate(query):
    # Retrieve relevant documents
    docs = st.session_state.vector_store.similarity_search(query, k=3)
    context = "\n".join(doc.page_content for doc in docs)
    
    # Generate response using Groq API
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        stream=True
    )
    
    # Stream response
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
    return full_response

# UI Components
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing files..."):
            raw_text = process_files(uploaded_files)
            chunks = split_text(raw_text)
            st.session_state.vector_store = create_vector_store(chunks)
        st.success(f"Processed {len(uploaded_files)} files")

# Chat interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask something about your documents"):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate and stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in retrieve_and_generate(user_query):
            full_response += chunk
            response_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})