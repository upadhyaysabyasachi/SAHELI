import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import pandas as pd

# -------------------- CONFIGURATION --------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
model = load_model()

# -------------------- LOAD PDFs --------------------
@st.cache_data
def load_chunks(path):
    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraph = " ".join(line.strip() for line in text.split("\n") if line.strip())
                chunks.append(paragraph)
    return chunks

anemia_chunks = load_chunks("data.pdf")
diabetes_chunks = load_chunks("diabetes.pdf")

# -------------------- LOAD EXCEL STP --------------------
@st.cache_data
def load_steps(path="STP_v2.xlsx"):
    xls = pd.ExcelFile(path)
    step_map = {
        "Anemia-Pregnant": xls.sheet_names[0],
        "Anemia-General": xls.sheet_names[1],
        "Diabetes-Pregnant": xls.sheet_names[2],
        "Diabetes-General": xls.sheet_names[3],
    }
    steps = {}
    for label, sheet in step_map.items():
        df = pd.read_excel(xls, sheet, skiprows=1)
        steps[label] = excel_to_text(df)
    return steps

def excel_to_text(df):
    text = []
    current_step = ""
    for _, row in df.iterrows():
        row = row.tolist()
        if pd.notna(row[0]) and "Step" in str(row[0]):
            current_step = f"{row[0]}: {row[1]}" if len(row) > 1 and pd.notna(row[1]) else row[0]
        elif len(row) > 2 and pd.notna(row[1]) and pd.notna(row[2]):
            text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return text

steps_context = load_steps()

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def create_embeddings():
    embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")

    condition_embeddings = {
        "anemia": embedder.encode(anemia_chunks[:10], convert_to_tensor=True),
        "diabetes": embedder.encode(diabetes_chunks[:10], convert_to_tensor=True)
    }

    all_steps_embeddings = {
        key: embedder.encode(value, convert_to_tensor=True)
        for key, value in steps_context.items() if value
    }

    return embedder, condition_embeddings, all_steps_embeddings

embedder, condition_embeddings, steps_embeddings = create_embeddings()

# -------------------- CLASSIFY CONDITION -------------------- based on cosine similarity of query with the respective guidelines of 
# anemia and diabetes
def classify_condition(prompt):
    prompt_embedding = embedder.encode(prompt, convert_to_tensor=True)
    anemia_score = util.pytorch_cos_sim(prompt_embedding, condition_embeddings["anemia"]).mean().item()
    diabetes_score = util.pytorch_cos_sim(prompt_embedding, condition_embeddings["diabetes"]).mean().item()
    return "anemia" if anemia_score >= diabetes_score else "diabetes"

# -------------------- RETRIEVAL FROM STP + PDF --------------------
def retrieve_relevant_chunks(prompt, selected_condition, top_k=5):
    query_embedding = embedder.encode(prompt, convert_to_tensor=True)

    sheet_key = (
        "Anemia-Pregnant" if selected_condition == "anemia" and "pregnant" in prompt.lower()
        else "Anemia-General" if selected_condition == "anemia"
        else "Diabetes-Pregnant" if "pregnant" in prompt.lower()
        else "Diabetes-General"
    )

    steps_text = steps_context.get(sheet_key, [])
    steps_tensor = steps_embeddings.get(sheet_key, None)
    pdf_chunks = anemia_chunks if selected_condition == "anemia" else diabetes_chunks
    pdf_tensor = embedder.encode(pdf_chunks, convert_to_tensor=True)

    results = []

    if steps_tensor is not None and steps_text:
        step_scores = util.pytorch_cos_sim(query_embedding, steps_tensor)[0]
        top_step_idx = torch.topk(step_scores, k=min(top_k, len(steps_text))).indices.tolist()
        results += [steps_text[i] for i in top_step_idx]

    pdf_scores = util.pytorch_cos_sim(query_embedding, pdf_tensor)[0]
    top_pdf_idx = torch.topk(pdf_scores, k=min(top_k, len(pdf_chunks))).indices.tolist()
    results += [pdf_chunks[i] for i in top_pdf_idx]

    return results

# -------------------- BUILD CONTEXT --------------------
def build_context(chat_history, chunks, condition):
    screening = [c for c in chunks if "Step" in c and " - Check " in c]
    guidelines = [c for c in chunks if c not in screening]
    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in chat_history)
    context = f"{history_text}\n\n"
    if screening:
        context += f"{condition.upper()} SCREENING PROTOCOL:\n" + "\n".join(screening) + "\n\n"
    if guidelines:
        context += f"{condition.upper()} GUIDELINES:\n" + "\n\n".join(guidelines)
    return context

# -------------------- SESSION INIT --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.write("Type a patient query or symptom. SAHELI will automatically detect the condition (anemia or diabetes) and assist accordingly.")

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- INPUT HANDLER --------------------
if prompt := st.chat_input("E.g. pregnant woman with RBS 200"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    selected_condition = classify_condition(prompt)
    chunks = retrieve_relevant_chunks(prompt, selected_condition)

    chat_pairs = [
        (q["content"], a["content"]) for q, a in zip(
            st.session_state.chat_history[::2], st.session_state.chat_history[1::2]
        ) if a["role"] == "assistant"
    ]

    context = build_context(chat_pairs, chunks, selected_condition)

    full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in anemia and diabetes detection, treatment, and management according to national guidelines.

Use the following information to generate a response:
{context}

User: {prompt}
Assistant:"""

    response = model.generate_content(full_prompt)
    answer = response.text.strip()

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -------------------- END SCREENING --------------------
st.markdown("---")
if st.button("🔁 End Screening"):
    for key in ["chat_history"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ Session ended. Ready for a new screening.")
    st.rerun()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("About SAHELI")
    st.write("""
SAHELI supports frontline workers in screening and managing anemia and diabetes based on national protocols.

✅ Auto-detects condition from user input  
📄 Uses STP and official PDF guidance  
🧠 Generates context-aware clinical advice  
""")
