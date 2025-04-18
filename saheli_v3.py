import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import pandas as pd

# -------------------- ENVIRONMENT SETUP --------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

model = load_model()

@st.cache_data
def load_chunks(path):
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                paragraph = " ".join(lines)
                chunks.append(paragraph)
    return chunks

anemia_chunks = load_chunks("data.pdf")
diabetes_chunks = load_chunks("diabetes.pdf")
all_chunks = anemia_chunks + diabetes_chunks

@st.cache_data
def load_all_excel_steps(excel_path='STP_v2.xlsx'):
    xls = pd.ExcelFile(excel_path)
    steps = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet, skiprows=1)
        steps[sheet] = excel_to_text(df)
    return steps

def excel_to_text(df):
    steps_text = []
    current_step = ""
    for _, row in df.iterrows():
        row = row.tolist()
        if not any(pd.notna(cell) for cell in row):
            continue
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            current_step = f"{row[0]}: {row[1]}" if len(row) > 1 and pd.notna(row[1]) else row[0]
        elif current_step:
            if len(row) >= 3 and pd.notna(row[1]) and pd.notna(row[2]):
                steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
            elif len(row) >= 2 and pd.notna(row[0]) and pd.notna(row[1]):
                steps_text.append(f"{current_step} - {row[0]}: {row[1]}")
    return steps_text

all_steps = load_all_excel_steps()

@st.cache_resource
def create_embeddings(text_chunks, step_dict):
    embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    step_embeddings = {
        k: embedder.encode(v, convert_to_tensor=True)
        for k, v in step_dict.items() if v
    }
    return embedder, pdf_embeddings, step_embeddings

embedder, corpus_embeddings, step_embeddings = create_embeddings(all_chunks, all_steps)

# -------------------- SESSION --------------------
if "pathway_selected" not in st.session_state:
    st.session_state.pathway_selected = False
if "selected_condition" not in st.session_state:
    st.session_state.selected_condition = ""
if "step_index" not in st.session_state:
    st.session_state.step_index = 0
if "step_responses" not in st.session_state:
    st.session_state.step_responses = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_completed" not in st.session_state:
    st.session_state.ai_completed = False

# -------------------- CONDITION SELECTION --------------------
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
            "Step 1: Are there any physical signs of anemia?",
            "Step 2: Any symptoms (fatigue, breathlessness)?",
            "Step 3: Hemoglobin value?",
            "Step 4: Has treatment started?"
        ],
        "Diabetes": [
            "Step 0: Is the person pregnant or not?",
            "Step 1: Fasting blood glucose level?",
            "Step 2: Post-meal glucose level?",
            "Step 3: Symptoms (thirst, urination)?",
            "Step 4: On insulin or oral meds?"
        ]
    }

    questions = stepwise_questions_map[selected_condition]
    st.subheader("📝 Collected Observations")
    for resp in st.session_state.step_responses:
        st.markdown(f"- {resp}")

    if st.session_state.step_index < len(questions):
        q = questions[st.session_state.step_index]
        answer = st.text_input(q, key=f"q{st.session_state.step_index}")
        if answer:
            st.session_state.step_responses.append(f"{q} — {answer}")
            st.session_state.chat_history.append({"role": "user", "content": answer})
            st.session_state.step_index += 1
            st.experimental_rerun()
    elif not st.session_state.ai_completed:
        user_summary = "\n".join(st.session_state.step_responses)
        steps_key = (
            "Anemia-Pregnant" if selected_condition == "Anemia" and "pregnant" in user_summary.lower() else
            "Anemia-General" if selected_condition == "Anemia" else
            "Diabetes-Pregnant" if "pregnant" in user_summary.lower() else
            "Diabetes-NonPregnant"
        )
        chunks = all_steps.get(steps_key, [])
        if chunks:
            q_embed = embedder.encode(user_summary, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(q_embed, step_embeddings[steps_key])[0]
            top_k = torch.topk(scores, k=min(5, len(chunks))).indices.tolist()
            relevant = [chunks[i] for i in top_k]
        else:
            relevant = []

        if selected_condition == "Anemia":
            relevant += anemia_chunks
        else:
            relevant += diabetes_chunks

        context = "\n".join(relevant)
        prompt = f"""You are SAHELI, a maternal health assistant chatbot for {selected_condition.lower()}.

Guidelines:
{context}

User Observations:
{user_summary}
"""

        response = model.generate_content(prompt)
        answer = response.text.strip()

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.ai_completed = True

    # ------------------ FOLLOW-UP CHAT ------------------
    if st.session_state.ai_completed:
        st.subheader("💬 Follow-Up Conversation")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        col1, col2 = st.columns([5, 1])
        with col1:
            followup = st.chat_input("Ask a follow-up question or type 'end' to finish")
        with col2:
            if st.button("🔁 End Screening"):
                followup = "end"

        if followup:
            with st.chat_message("user"):
                st.markdown(followup)

            if followup.strip().lower() == "end":
                for key in ["pathway_selected", "selected_condition", "step_index", "step_responses", "chat_history", "ai_completed"]:
                    st.session_state.pop(key, None)
                st.success("✅ Session ended. Ready for a new patient.")
                st.experimental_rerun()
            else:
                st.session_state.chat_history.append({"role": "user", "content": followup})
                history_text = "\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state.chat_history
                ])
                followup_prompt = f"""You are SAHELI, continuing a maternal health consultation.

Previous conversation:
{history_text}

Respond with clear clinical guidance as per the protocols."""

                reply = model.generate_content(followup_prompt).text.strip()

                with st.chat_message("assistant"):
                    st.markdown(reply)

                st.session_state.chat_history.append({"role": "assistant", "content": reply})
