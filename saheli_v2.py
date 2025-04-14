
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import json

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant for Anemia Detection")

genai.configure(api_key="AIzaSyDJCxJZhceUiN__d1JO_Ha-N2o9v6Sf6Pg")

@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

model = load_gemini_model()

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_json_protocol(json_path='enhanced_anemia_protocol.json'):
    with open(json_path) as file:
        return json.load(file)

anemia_protocol = load_json_protocol()

@st.cache_data
def load_chunks(pdf_path='data.pdf'):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraph = ' '.join([line.strip() for line in text.split('\n') if line.strip()])
                chunks.append(paragraph)
    return chunks

chunks = load_chunks()

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def create_embeddings(text_chunks):
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    return embedder, embedder.encode(text_chunks, convert_to_tensor=True)

embedder, corpus_embeddings = create_embeddings(chunks)

def retrieve_chunks(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(chunks)))
    return [chunks[idx] for idx in top_results[1].tolist()]

# -------------------- CONTEXT BUILDER --------------------
def build_context(chat_history, retrieved_chunks, anemia_protocol):
    signs_text = "\n".join([f"{k}: {v['description']}" for k, v in anemia_protocol["Step 1"]["Signs"].items()])
    grading_text = "\n".join([f"{grade['Hb (gm/dl)']} ({grade['Grade']}): {grade['Action'] if 'Action' in grade else 'Refer conditions'}" for grade in anemia_protocol["Step 4"]["Grading and Management"]])
    guidelines_text = "\n\n".join(retrieved_chunks)
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
    
    context = f"{history_text}\n\nANEMIA SIGNS (Step 1):\n{signs_text}\n\nANEMIA GRADING (Step 4):\n{grading_text}\n\nGUIDELINES:\n{guidelines_text}\n"
    return context

# -------------------- SESSION STATE --------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# -------------------- MAIN INTERFACE --------------------
st.write("Describe your patient's condition for guideline-based advice.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter patient details here"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    retrieved_chunks = retrieve_chunks(prompt)
    context = build_context([(q["content"], a["content"]) for q, a in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]) if a["role"] == "assistant"], retrieved_chunks, anemia_protocol)

    full_prompt = f"""You are SAHELI, specialized in anemia management per Anemia Mukt Bharat.

    Steps:
    1. Check STEP 1 signs.
    2. If any sign is positive, instruct Hb check (STEP 3).
    3. Classify Hb and recommend actions (STEP 4).
    
    Context:
    {context}
    
    Query: {prompt}
    SAHELI Response:"""

    response = model.generate_content(full_prompt)
    answer = response.text.strip()

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# -------------------- SIDEBAR INTERACTIVE WIZARD --------------------
with st.sidebar:
    st.header("Anemia Signs Assessment (Step 1)")
    signs_results = {}
    for sign, details in anemia_protocol["Step 1"]["Signs"].items():
        signs_results[sign] = st.radio(f"{sign}: {details['description']}", ["No", "Yes"])

    if "Yes" in signs_results.values():
        st.success("Signs positive: Proceed to Hb check (Step 3).")
        hb_val = st.number_input("Step 3: Hb Level (gm/dL)", 0.0, 20.0, step=0.1)
        if st.button("Step 4 Recommendations"):
            st.session_state.hb_val = hb_val
            st.experimental_rerun()
    else:
        st.info("No signs detected; no immediate Hb check required.")

    if 'hb_val' in st.session_state:
        hb = st.session_state.hb_val
        recommendation = next((rec for rec in anemia_protocol["Step 4"]["Grading and Management"] if (
            ('-' in rec["Hb (gm/dl)"] and float(rec["Hb (gm/dl)"].split('-')[0]) <= hb <= float(rec["Hb (gm/dl)"].split('-')[1])) or
            ('>' in rec["Hb (gm/dl)"] and hb > float(rec["Hb (gm/dl)"].strip('>'))) or
            ('<' in rec["Hb (gm/dl)"] and hb < float(rec["Hb (gm/dl)"].strip('<'))))), None)

        if recommendation:
            st.write(f"**{recommendation['Grade']} Anaemia**: {recommendation['Action']}")
        else:
            st.error("Please verify Hb input.")
