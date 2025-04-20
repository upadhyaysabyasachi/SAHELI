import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import os
import pandas as pd
import time

# -------------------- ENVIRONMENT SETUP --------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

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

# Load both PDFs and combine their chunks
anemia_chunks = load_chunks('data.pdf')
diabetes_chunks = load_chunks('diabetes.pdf')
chunks = anemia_chunks + diabetes_chunks


@st.cache_data
def load_all_excel_steps(excel_path='STP_v2.xlsx'):
    try:
        xls = pd.ExcelFile(excel_path)
        steps = {}
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet, skiprows=1)
            steps[sheet] = excel_to_text(df)
        return steps
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}")
        return {}

def excel_to_text(df):
    steps_text = []
    current_step = ""
    for index, row in df.iterrows():
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            current_step = row[0] + ": " + str(row[1]) if pd.notna(row[1]) else row[0]
        elif pd.notna(row[1]) and pd.notna(row[2]):
            steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return steps_text

all_steps = load_all_excel_steps()

@st.cache_resource
def create_embeddings(text_chunks, all_steps):
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    step_embeddings = {k: embedder.encode(v, convert_to_tensor=True) for k, v in all_steps.items()}
    return embedder, pdf_embeddings, step_embeddings

embedder, corpus_embeddings, step_embeddings = create_embeddings(chunks, all_steps)

# -------------------- PATHWAY SELECTION --------------------
if 'pathway_selected' not in st.session_state:
    st.session_state.pathway_selected = False
if 'selected_condition' not in st.session_state:
    st.session_state.selected_condition = ""

if not st.session_state.pathway_selected:
    condition = st.radio("Which condition are you screening for?", ["Anemia", "Diabetes"])
    if st.button("Start Screening"):
        st.session_state.selected_condition = condition
        st.session_state.pathway_selected = True
        st.experimental_rerun()
else:
    selected_condition = st.session_state.selected_condition

    stepwise_questions_map = {
        "Anemia": [
            "Step 0: Is the woman pregnant or not?",
            "Step 1: Are there any physical signs of anemia (pale eyelids, palms, tongue, or brittle nails)?",
            "Step 2: Is the woman experiencing any symptoms (dizziness, tiredness, fast heartbeat, breathlessness)?",
            "Step 3: What is the hemoglobin value (if known)?",
            "Step 4: Based on severity and gestation, has any treatment been started?"
        ],
        "Diabetes": [
            "Step 0: Is the person pregnant or not?",
            "Step 1: What is the fasting blood glucose level?",
            "Step 2: What is the postprandial blood glucose level?",
            "Step 3: Are there any symptoms (increased thirst, urination, fatigue)?",
            "Step 4: Any medication or insulin being used currently?"
        ]
    }

    if 'step_index' not in st.session_state:
        st.session_state.step_index = 0
    if 'step_responses' not in st.session_state:
        st.session_state.step_responses = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    stepwise_questions = stepwise_questions_map[selected_condition]

    # Display collected responses
    st.subheader("📝 Collecting Observations")
    for step in st.session_state.step_responses:
        st.markdown(f"- {step}")

    if st.session_state.step_index < len(stepwise_questions):
        current_question = stepwise_questions[st.session_state.step_index]
        user_input = st.text_input(current_question, key=f"step_{st.session_state.step_index}")

        if user_input:
            st.session_state.step_responses.append(f"{current_question} — Response: {user_input}")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.step_index += 1
            st.experimental_rerun()

    else:
        user_summary = "\n".join(st.session_state.step_responses)
        prompt = f"Proceed with {selected_condition.lower()} diagnosis based on observations."

        # Decide which sheets to retrieve from
        if selected_condition == "Anemia":
            if "pregnant" in user_summary.lower():
                steps_key = "Anemia-Pregnant"
            else:
                steps_key = "Anemia-NonPregnant"
        else:  # Diabetes
            if "pregnant" in user_summary.lower():
                steps_key = "Diabetes-Pregnant"
            else:
                steps_key = "Diabetes-NonPregnant"

        relevant_chunks = all_steps.get(steps_key, [])
        embeddings = step_embeddings.get(steps_key)
        if embeddings is not None:
            query_embedding = embedder.encode(user_summary, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_indices = torch.topk(scores, k=min(5, len(relevant_chunks))).indices.tolist()
            retrieved_chunks = [relevant_chunks[i] for i in top_indices]
        else:
            retrieved_chunks = []

        context = "\n".join(retrieved_chunks)
        full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in {selected_condition.lower()} detection, screening, and follow-up based on national health protocols and attached pdfs added as context.

You must:
1. Provide specific, actionable advice based strictly on official Anemia Mukt Bharat guidelines
2. Follow the structured screening protocol for detection of anemia
3. Recommend appropriate tests, treatments, and follow-ups based on evidence
4. Use simple, clear language appropriate for healthcare workers in rural India
5. Be concise but thorough in your explanations
6. Never invent symptoms, treatments, or recommendations not supported by the provided context
       
Here are the observations:
{user_summary}

Relevant Guidelines:
{context}

User: {prompt}
Assistant:"""

        response = model.generate_content(full_prompt)
        answer = response.text.strip()

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with st.sidebar:
        st.header("About SAHELI")
        st.write(f"""
        SAHELI helps healthcare workers screen, diagnose, and manage {selected_condition.lower()} in the field using national protocols.

        This tool supports:
        - Step-by-step screening
        - Clinical decision support
        - Treatment planning
        - Follow-up suggestions
        """)


