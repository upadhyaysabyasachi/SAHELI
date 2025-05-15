import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import pandas as pd

# -------------------- ENVIRONMENT SETUP --------------------
# Configure Gemini API with secret key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

model = load_gemini_model()

# -------------------- LOAD PDF --------------------
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
 

# -------------------- LOAD EXCEL --------------------
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
    for _, row in df.iterrows():
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            current_step = row[0] + ": " + str(row[1]) if pd.notna(row[1]) else row[0]
        elif pd.notna(row[1]) and pd.notna(row[2]):
            steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return steps_text

all_steps = load_all_excel_steps()

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def create_embeddings(text_chunks, all_steps):
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    step_embeddings = {
        k: embedder.encode(v, convert_to_tensor=True)
        for k, v in all_steps.items() if v
    }
    return embedder, pdf_embeddings, step_embeddings

embedder, corpus_embeddings, step_embeddings = create_embeddings(chunks, all_steps)

# -------------------- CONDITION SELECTION --------------------
if 'pathway_selected' not in st.session_state:
    st.session_state.pathway_selected = False
if 'selected_condition' not in st.session_state:
    st.session_state.selected_condition = ""

if not st.session_state.pathway_selected:
    condition = st.radio("Which condition are you screening for?", ["Anemia", "Diabetes"])
    if st.button("Start Screening"):
        st.session_state.selected_condition = condition
        st.session_state.pathway_selected = True
        st.rerun()
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

    st.subheader("📝 Collected Observations")
    for step in st.session_state.step_responses:
        st.markdown(f"- {step}")

    if st.session_state.step_index < len(stepwise_questions):
        current_question = stepwise_questions[st.session_state.step_index]
        user_input = st.text_input(current_question, key=f"step_{st.session_state.step_index}")
        if user_input:
            st.session_state.step_responses.append(f"{current_question} — Response: {user_input}")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.step_index += 1
            st.rerun()

    else:
        user_summary = "\n".join(st.session_state.step_responses)
        prompt = f"Proceed with {selected_condition.lower()} diagnosis based on observations."

        if selected_condition == "Anemia":
            steps_key = "anemia-pregnant" if "pregnant" in user_summary.lower() else "anemia-general"
        else:
            steps_key = "diabetes-pregnant" if "pregnant" in user_summary.lower() else "diabetes-general"

        relevant_chunks = all_steps.get(steps_key, [])
        embeddings = step_embeddings.get(steps_key)

        if embeddings:
            query_embedding = embedder.encode(user_summary, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_indices = torch.topk(scores, k=min(5, len(relevant_chunks))).indices.tolist()
            retrieved_chunks = [relevant_chunks[i] for i in top_indices]
        else:
            retrieved_chunks = []

        context_chunks = retrieved_chunks
        if selected_condition == "Anemia":
            context_chunks += chunks

        context = "\n".join(context_chunks)

        full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in {selected_condition.lower()} detection, screening, and follow-up based on national health protocols.

Generate an imagery wherever dietary recommendations as a part of the response.

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

        # -------------------- FOLLOW-UP SUPPORT --------------------
        st.subheader("💬 Continue the Conversation")
        col1, col2 = st.columns([3, 1])
        with col1:
            followup_prompt = st.chat_input("Ask a follow-up question or type 'end' to finish")
        with col2:
            if st.button("🔁 End Screening"):
                followup_prompt = "end"

        if followup_prompt:
            if followup_prompt.strip().lower() == "end":
                for key in ["pathway_selected", "selected_condition", "step_index", "step_responses", "chat_history"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Screening ended. Ready for the next patient.")
                st.rerun()
            else:
                st.session_state.chat_history.append({"role": "user", "content": followup_prompt})

                history_text = "\n".join([
                    f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                    for m in st.session_state.chat_history
                ])
                continued_prompt = f"""You are SAHELI, continuing a medical support conversation for {selected_condition.lower()} with a healthcare worker.

Conversation so far:
{history_text}

Respond to the latest user query with appropriate clinical guidance, referring to earlier context and official protocol if needed."""

                continued_response = model.generate_content(continued_prompt)
                continued_answer = continued_response.text.strip()

                with st.chat_message("assistant"):
                    st.markdown(continued_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": continued_answer})

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
