import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import os

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="SAHELI Chatbot", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Chatbot")

# Configure Gemini API
genai.configure(api_key="AIzaSyDJCxJZhceUiN__d1JO_Ha-N2o9v6Sf6Pg")  # Replace with your actual API key

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

model = load_gemini_model()

# -------------------- LOAD PDF DATA & CREATE CHUNKS --------------------
@st.cache_data
def load_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                paragraph = ' '.join(lines)
                chunks.append(paragraph)
    return chunks

chunks = load_chunks('data.pdf')

# -------------------- BUILD VECTOR STORE --------------------
@st.cache_resource
def create_embeddings(text_chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    return embedder, embeddings

embedder, corpus_embeddings = create_embeddings(chunks)

# -------------------- RETRIEVAL FUNCTION --------------------
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [chunks[idx] for idx in top_results[1].tolist()]

# -------------------- CONTEXT BUILDER --------------------
def build_context(chat_history, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
    return f"{history_text}\n\nContext:\n{context}"

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -------------------- MAIN CHAT INTERFACE --------------------
st.write("Feel free to ask any questions about maternal healthcare!")

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question about maternal healthcare:"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Retrieve relevant context
    relevant_chunks = retrieve_relevant_chunks(prompt)

    # Build prompt for Gemini
    context = build_context(st.session_state.chat_history, relevant_chunks)
    full_prompt = f"""You are SAHELI, a helpful maternal healthcare assistant chatbot.

Here is the conversation so far:
{context}

User: {prompt}
Assistant:"""

    # Get Gemini response
    response = model.generate_content(full_prompt)
    answer = response.text.strip()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
