import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import os
import pandas as pd
#from dotenv import load_dotenv

# Load environment variables
#load_dotenv()

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="SAHELI Chatbot", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Chatbot for Anemia Detection")

# Configure Gemini API
#"AIzaSyDJCxJZhceUiN__d1JO_Ha-N2o9v6Sf6Pg"
genai.configure(api_key="AIzaSyDJCxJZhceUiN__d1JO_Ha-N2o9v6Sf6Pg")

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

# -------------------- LOAD EXCEL ANEMIA SCREENING PROTOCOL DATA --------------------
@st.cache_data
def load_excel_steps(excel_path='AnaemiaSTP.xlsx'):
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_df = pd.read_excel(xls, 'Sheet1', skiprows=5)  # Adjust row skip based on actual content
        return excel_to_text(sheet_df)
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}. Using default screening steps.")
        # Providing default screening steps in case the Excel file isn't available
        return [
            "Step 1: Physical signs - Check for pale lower eyelids: If inner eyelids appear pale, this indicates possible anemia",
            "Step 1: Physical signs - Check for pale tongue: If tongue appears pale, this indicates possible anemia",
            "Step 1: Physical signs - Check for pale skin: If skin appears pale, this indicates possible anemia",
            "Step 1: Physical signs - Check for pale palms: If palms appear pale, this indicates possible anemia",
            "Step 1: Physical signs - Check for brittle nails: If nails are brittle, this indicates possible anemia",
            "Step 2: Symptoms - Ask about dizziness: If patient reports dizziness, this indicates possible anemia",
            "Step 2: Symptoms - Ask about unusual tiredness: If patient reports unusual fatigue, this indicates possible anemia",
            "Step 2: Symptoms - Ask about rapid heart rate: If patient reports heart palpitations, this indicates possible anemia",
            "Step 2: Symptoms - Ask about shortness of breath: If patient reports difficulty breathing, this indicates possible anemia"
        ]

def excel_to_text(df):
    steps_text = []
    current_step = ""
    
    for index, row in df.iterrows():
        # Check if this is a step header
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            current_step = row[0] + ": " + str(row[1]) if pd.notna(row[1]) else row[0]
        # Check if this is a detailed item 
        elif pd.notna(row[1]) and pd.notna(row[2]):
            steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    
    return steps_text

# Load Excel steps
steps_context = load_excel_steps()

# -------------------- BUILD VECTOR STORE --------------------
@st.cache_resource
def create_embeddings(text_chunks, steps_text):
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    # Create embeddings for PDF chunks
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    # Create embeddings for Excel steps
    steps_embeddings = embedder.encode(steps_text, convert_to_tensor=True)
    return embedder, pdf_embeddings, steps_embeddings

embedder, corpus_embeddings, steps_embeddings = create_embeddings(chunks, steps_context)

# -------------------- ENHANCED RETRIEVAL FUNCTION --------------------
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Check if query is related to screening or symptoms
    anemia_screening_keywords = ["screening", "signs", "symptoms", "check", "detect", "detection", 
                              "pale", "diagnosis", "diagnose", "test", "testing", "examine"]
    
    # Get scores from both knowledge bases
    pdf_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    steps_scores = util.pytorch_cos_sim(query_embedding, steps_embeddings)[0]
    
    if any(keyword in query.lower() for keyword in anemia_screening_keywords):
        # Prioritize screening steps but include some PDF content too
        steps_top_k = min(top_k, len(steps_context))
        steps_top_results = torch.topk(steps_scores, k=steps_top_k)
        steps_results = [steps_context[idx] for idx in steps_top_results[1].tolist()]
        
        # Add some PDF content if needed to reach top_k
        if steps_top_k < top_k:
            pdf_top_results = torch.topk(pdf_scores, k=(top_k - steps_top_k))
            pdf_results = [chunks[idx] for idx in pdf_top_results[1].tolist()]
            return steps_results + pdf_results
        return steps_results
    else:
        # For general queries, prioritize PDF content but include relevant screening steps
        pdf_top_results = torch.topk(pdf_scores, k=min(top_k, len(chunks)))
        pdf_results = [chunks[idx] for idx in pdf_top_results[1].tolist()]
        
        # Check if there are any highly relevant screening steps
        # Get top screening step if its score is high enough
        if len(steps_scores) > 0:
            max_step_score, max_step_idx = torch.max(steps_scores, dim=0)
            if max_step_score > 0.5:  # If relevance score is significant
                best_step = steps_context[max_step_idx.item()]
                if len(pdf_results) >= top_k:
                    pdf_results[-1] = best_step  # Replace least relevant PDF result
                else:
                    pdf_results.append(best_step)
        
        return pdf_results

# -------------------- ENHANCED CONTEXT BUILDER --------------------
def build_context(chat_history, retrieved_chunks):
    # Separate screening steps from guidelines in the retrieved chunks
    screening_steps = []
    guidelines = []
    
    for chunk in retrieved_chunks:
        if "Step" in chunk and " - Check " in chunk:
            screening_steps.append(chunk)
        else:
            guidelines.append(chunk)
    
    # Build context sections
    screening_text = "\n".join(screening_steps) if screening_steps else ""
    guidelines_text = "\n\n".join(guidelines) if guidelines else ""
    
    # Format chat history
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
    
    # Create the final context with clear sections
    context = f"{history_text}\n\n"
    
    if screening_text:
        context += f"ANEMIA SCREENING PROTOCOL:\n{screening_text}\n\n"
    
    if guidelines_text:
        context += f"ANEMIA MUKT BHARAT GUIDELINES:\n{guidelines_text}\n\n"
    
    return context

# -------------------- SESSION STATE INITIALIZATION --------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -------------------- MAIN CHAT INTERFACE --------------------
st.write("Mention the condition of your patient, and we will help you with the best possible advice based on approved guidelines")

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Put the details of your patient here"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Retrieve relevant context
    relevant_chunks = retrieve_relevant_chunks(prompt)

    # Build context for Gemini with enhanced structure
    context = build_context(
        [(q["content"], a["content"]) for q, a in zip(
            st.session_state.chat_history[::2], 
            st.session_state.chat_history[1::2]
        ) if a["role"] == "assistant"],
        relevant_chunks
    )

    # Enhanced prompt with specialization for anemia
    full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in anemia detection, treatment, and management according to the Anemia Mukt Bharat (AMB) guidelines.

You must:
1. Provide specific, actionable advice based strictly on official Anemia Mukt Bharat guidelines
2. Follow the structured screening protocol for detection of anemia
3. Recommend appropriate tests, treatments, and follow-ups based on evidence
4. Use simple, clear language appropriate for healthcare workers in rural India
5. Be concise but thorough in your explanations
6. Never invent symptoms, treatments, or recommendations not supported by the provided context

Here is relevant context to inform your response:
{context}

User: {prompt}
Assistant (Provide guideline-aligned response about anemia management):"""

    # Get Gemini response with expanded context
    response = model.generate_content(full_prompt)
    answer = response.text.strip()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -------------------- SIDEBAR WITH INFORMATION --------------------
with st.sidebar:
    st.header("About SAHELI Anemia Detection")
    st.write("""
    SAHELI helps healthcare workers screen, diagnose, and manage anemia in women according to the Anemia Mukt Bharat (AMB) guidelines.
    
    This tool supports:
    - Step-by-step anemia screening protocols
    - Clinical decision support
    - Treatment guidelines
    - Follow-up recommendations
    """)
    
    st.header("Key Screening Steps")
    st.write("""
    **Step 1: Physical Signs**
    - Check for pale lower eyelids, tongue, skin, palms
    - Check for brittle nails
    
    **Step 2: Symptoms**
    - Ask about dizziness, unusual tiredness
    - Ask about rapid heart rate, shortness of breath
    
    **Step 3: Testing**
    - Hemoglobin estimation
    - Classification by severity
    """)
