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
st.title("🤖 SAHELI: Maternal Healthcare Assistant for Anemia Detection")

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

@st.cache_data
def load_excel_steps(excel_path='AnaemiaSTP.xlsx'):
    try:
        xls = pd.ExcelFile(excel_path)
        sheet1_df = pd.read_excel(xls, 'Sheet1', skiprows=1)
        sheet2_df = pd.read_excel(xls, 'Sheet2', skiprows=1)
        sheet_df = pd.concat([sheet1_df, sheet2_df], ignore_index=True)
        return excel_to_text(sheet_df)
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}. Using default screening steps.")
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
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            current_step = row[0] + ": " + str(row[1]) if pd.notna(row[1]) else row[0]
        elif pd.notna(row[1]) and pd.notna(row[2]):
            steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return steps_text

steps_context = load_excel_steps()

@st.cache_resource
def create_embeddings(text_chunks, steps_text):
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    steps_embeddings = embedder.encode(steps_text, convert_to_tensor=True)
    return embedder, pdf_embeddings, steps_embeddings

embedder, corpus_embeddings, steps_embeddings = create_embeddings(chunks, steps_context)

def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    anemia_keywords = ["screening", "signs", "symptoms", "check", "detect", "detection", 
                       "pale", "diagnosis", "diagnose", "test", "testing", "examine"]
    pdf_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    steps_scores = util.pytorch_cos_sim(query_embedding, steps_embeddings)[0]
    if any(keyword in query.lower() for keyword in anemia_keywords):
        steps_top_k = min(top_k, len(steps_context))
        steps_top_results = torch.topk(steps_scores, k=steps_top_k)
        steps_results = [steps_context[idx] for idx in steps_top_results[1].tolist()]
        if steps_top_k < top_k:
            pdf_top_results = torch.topk(pdf_scores, k=(top_k - steps_top_k))
            pdf_results = [chunks[idx] for idx in pdf_top_results[1].tolist()]
            return steps_results + pdf_results
        return steps_results
    else:
        pdf_top_results = torch.topk(pdf_scores, k=min(top_k, len(chunks)))
        pdf_results = [chunks[idx] for idx in pdf_top_results[1].tolist()]
        if len(steps_scores) > 0:
            max_step_score, max_step_idx = torch.max(steps_scores, dim=0)
            if max_step_score > 0.5:
                best_step = steps_context[max_step_idx.item()]
                if len(pdf_results) >= top_k:
                    pdf_results[-1] = best_step
                else:
                    pdf_results.append(best_step)
        return pdf_results

def build_context(chat_history, retrieved_chunks):
    screening_steps = []
    guidelines = []
    for chunk in retrieved_chunks:
        if "Step" in chunk and " - Check " in chunk:
            screening_steps.append(chunk)
        else:
            guidelines.append(chunk)
    screening_text = "\n".join(screening_steps) if screening_steps else ""
    guidelines_text = "\n\n".join(guidelines) if guidelines else ""
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
    context = f"{history_text}\n\n"
    if screening_text:
        context += f"ANEMIA SCREENING PROTOCOL:\n{screening_text}\n\n"
    if guidelines_text:
        context += f"ANEMIA MUKT BHARAT GUIDELINES:\n{guidelines_text}\n\n"
    return context

# -------------------- STEPWISE INTERACTION --------------------
stepwise_questions = [
    "Step 0: Is the woman pregnant or not?",
    "Step 1: Are there any physical signs of anemia (pale eyelids, palms, tongue, or brittle nails)?",
    "Step 2: Is the woman experiencing any symptoms (dizziness, tiredness, fast heartbeat, breathlessness)?",
    "Step 3: What is the hemoglobin value (if known)?",
    "Step 4: Based on severity and gestation, has any treatment been started?"
]

if 'step_index' not in st.session_state:
    st.session_state.step_index = 0
if 'step_responses' not in st.session_state:
    st.session_state.step_responses = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display previously entered responses so far
st.subheader("📝 Collected Observations")
for step in st.session_state.step_responses:
    st.markdown(f"- {step}")

# Display next question only if in stepwise phase
if st.session_state.step_index < len(stepwise_questions):
    current_question = stepwise_questions[st.session_state.step_index]
    user_input = st.text_input(current_question, key=f"step_{st.session_state.step_index}")

    if user_input:
        st.session_state.step_responses.append(f"{current_question} — Response: {user_input}")
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.step_index += 1
        st.experimental_rerun()

# Once all steps are answered
else:
    user_summary = "\n".join(st.session_state.step_responses)
    prompt = "Proceed with anemia diagnosis based on observations."
    relevant_chunks = retrieve_relevant_chunks(user_summary)
    context = build_context(
        [(q["content"], a["content"]) for q, a in zip(
            st.session_state.chat_history[::2], 
            st.session_state.chat_history[1::2]
        ) if a["role"] == "assistant"],
        relevant_chunks
    )

    full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in anemia detection, treatment, and management according to the Anemia Mukt Bharat (AMB) guidelines.
     This will be used by a health worker based out of India for screening, detection and treatment.

Follow this 5-step procedure based on the standard screening protocol from the Anemia Screening & Treatment Pathway (AnemiaSTP):

**Step 0: Ask whether she is pregnant ?**
    - If pregnant, then refer to 'Sheet 1' workflow
    - If not pregnant, then refer to 'Sheet 2' workflow 

**Step 1: Physical Signs**
- Check for visible signs of pallor (lower eyelids, tongue, skin, palms), and brittle nails.
- If any are present, proceed to Step 2.

**Step 2: Symptoms**
- Ask about dizziness, fatigue, rapid heartbeat, or shortness of breath.
- If any symptoms are present (with or without Step 1 signs), proceed to Step 3.

**Step 3: Hemoglobin Testing**
- Recommend Hb testing using a digital hemoglobinometer.
- Use the value to classify anemia by severity as per guidelines.

**Step 4: Treatment Action**
- Based on anemia grading and gestational age, recommend treatment using IFA, IV Iron, or hospital referral.
- Always align with the trimester-based action plan.

Expose the above steps at the beginning only for the healthcare worker to respond.

You must:
1. Provide specific, actionable advice based strictly on official Anemia Mukt Bharat guidelines
2. Follow the structured screening protocol for detection of anemia
3. Recommend appropriate tests, treatments, and follow-ups based on evidence
4. Use simple, clear language appropriate for healthcare workers in rural India
5. Be concise but thorough in your explanations
6. Never invent symptoms, treatments, or recommendations not supported by the provided context

Follow the official 5-step procedure:

{user_summary}

Here is relevant context to inform your response:
{context}

User: {prompt}
Assistant:"""

    response = model.generate_content(full_prompt)
    answer = response.text.strip()

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

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
