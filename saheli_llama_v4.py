import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch, pdfplumber, pandas as pd, logging, requests, json
#import google.generativeai as genai
import pickle, argparse, pdfplumber, pandas as pd, torch
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
API_URL = "https://cloud.olakrutrim.com/v1/chat/completions"
MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"
BEARER_TOKEN = st.secrets["BEARER_TOKEN"]        # <-- env-var in production

#logging.basicConfig(
#    level=logging.DEBUG,
#    format="%(asctime)s | %(levelname)s | %(message)s",
#    force=True,
#)
log = logging.getLogger("SAHELI")

#genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
log.info("SAHELI app started. Initializing configurations.")


st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

# --------------------------------------------------------------------
# PDF LOADER
# --------------------------------------------------------------------
@st.cache_data
def load_chunks(path: str):
    """Return a list of paragraph strings from a PDF file."""
    chunks = []
    with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    paragraph = " ".join(line.strip() for line in text.split("\n") if line.strip())
                    chunks.append(paragraph)
    return chunks

log.info("Loading PDF chunks from data files.")

anemia_chunks    = load_chunks("data.pdf")
diabetes_chunks  = load_chunks("diabetes.pdf")
nutrition_chunks = load_chunks("icds_operational_guidelines_for_wifs.pdf")

log.info("Loading of chunks completed")

all_pdf_chunks = {
    "anemia": anemia_chunks,
    "diabetes": diabetes_chunks,
    "nutrition": nutrition_chunks,
}

# --------------------------------------------------------------------
# EXCEL STP LOADER
# --------------------------------------------------------------------
log.info("Loading STP steps from Excel file.")

@st.cache_data
def load_steps(path: str = "STP_v2.xlsx"):
    xls = pd.ExcelFile(path)
    step_map = {
        "Anemia-Pregnant":  xls.sheet_names[0],
        "Anemia-General":   xls.sheet_names[1],
        "Diabetes-Pregnant": xls.sheet_names[2],
        "Diabetes-General":  xls.sheet_names[3],
    }
    steps = {}
    for label, sheet in step_map.items():
        df = pd.read_excel(xls, sheet, skiprows=1)
        steps[label] = excel_to_text(df)
    return steps

def excel_to_text(df: pd.DataFrame):
    text, current_step = [], ""
    for _, row in df.iterrows():
        row = row.tolist()
        if pd.notna(row[0]) and "Step" in str(row[0]):
            current_step = f"{row[0]}: {row[1]}" if len(row) > 1 and pd.notna(row[1]) else row[0]
        elif len(row) > 2 and pd.notna(row[1]) and pd.notna(row[2]):
            text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return text

steps_context = load_steps()

# --------------------------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------------------------
log.info("Creating sentence embeddings using BioBERT.")

@st.cache_resource(show_spinner="🔄 Loading pre-built embeddings …")
def load_static_embeddings(pkl_path: str = "embeddings.pkl"):
    blob = pickle.load(open(pkl_path, "rb"))

    model_id = blob.get(
        "model_name",
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )
    
    embedder = SentenceTransformer(model_id)  # <-- use model_name now

    condition_emb     = blob["condition_embeddings"]
    steps_emb         = blob["steps_embeddings"]
    nutrition_emb     = blob["nutrition_embeddings"]

    return (embedder,
            condition_emb,
            steps_emb,
            nutrition_emb,
            blob["steps_context"],
            blob["condition_chunks"],
            blob["nutrition_chunks"])


(embedder,
 condition_embeddings,
 steps_embeddings,
 nutrition_embeddings,
 steps_context,
 condition_chunks,
 nutrition_chunks) = load_static_embeddings()
# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
NUTRITION_KEYS = [
    "diet", "food", "foods", "nutrition", "meal", "menu",
    "iron rich", "vitamin c", "wifs", "balanced diet",
]

def needs_nutrition(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in NUTRITION_KEYS)

def classify_condition(prompt: str) -> str:
    prompt_emb = embedder.encode(prompt, convert_to_tensor=True)
    anemia_score   = util.pytorch_cos_sim(prompt_emb, condition_embeddings["anemia"]).mean().item()
    diabetes_score = util.pytorch_cos_sim(prompt_emb, condition_embeddings["diabetes"]).mean().item()
    return "anemia" if anemia_score >= diabetes_score else "diabetes"

def retrieve_relevant_chunks(prompt: str, condition: str, top_k: int = 5):
    query_emb = embedder.encode(prompt, convert_to_tensor=True)

    sheet_key = (
        "Anemia-Pregnant"   if condition == "anemia"   and "pregnant" in prompt.lower() else
        "Anemia-General"    if condition == "anemia"   else
        "Diabetes-Pregnant" if "pregnant" in prompt.lower() else
        "Diabetes-General"
    )

    hits = []

    # STP hits
    stp_text   = steps_context.get(sheet_key, [])
    stp_tensor = steps_embeddings.get(sheet_key)
    if stp_tensor is not None and stp_text:
        stp_scores = util.pytorch_cos_sim(query_emb, stp_tensor)[0]
        stp_idx = torch.topk(stp_scores, k=min(top_k, len(stp_text))).indices.tolist()
        hits.extend(stp_text[i] for i in stp_idx)

    # PDF hits
    pdf_chunks = all_pdf_chunks[condition]
    pdf_tensor = embedder.encode(pdf_chunks, convert_to_tensor=True)
    pdf_scores = util.pytorch_cos_sim(query_emb, pdf_tensor)[0]
    pdf_idx    = torch.topk(pdf_scores, k=min(top_k, len(pdf_chunks))).indices.tolist()
    hits.extend(pdf_chunks[i] for i in pdf_idx)

    # Nutrition hits (optional)
    if needs_nutrition(prompt):
        n_scores = util.pytorch_cos_sim(query_emb, nutrition_embeddings)[0]
        n_idx    = torch.topk(n_scores, k=min(3, len(nutrition_chunks))).indices.tolist()
        hits.extend(nutrition_chunks[i] for i in n_idx)

    return hits

def build_context(history_pairs, chunks, condition):
    screening  = [c for c in chunks if "Step" in c and " - Check " in c]
    guidelines = [c for c in chunks if c not in screening]
    ctx = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history_pairs) + "\n\n"
    if screening:
        ctx += f"{condition.upper()} SCREENING PROTOCOL:\n" + "\n".join(screening) + "\n\n"
    if guidelines:
        ctx += f"{condition.upper()} GUIDELINES:\n" + "\n\n".join(guidelines)
    return ctx

# --------------------------------------------------------------------
# STREAMLIT SESSION
# --------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.write(
    "Type a patient query or symptom. SAHELI will automatically detect "
    "the condition (anemia or diabetes) and assist accordingly."
)

# replay prior messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------
# MAIN INPUT HANDLER
# --------------------------------------------------------------------
if prompt := st.chat_input("E.g. pregnant woman with RBS 200"):
    # display user msg
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # retrieve context
    log.info("Initializing classification of condition")
    condition   = classify_condition(prompt)
    log.info("Condition is classified")
    log.info("Retrieving the relevant chunks")
    chunks      = retrieve_relevant_chunks(prompt, condition)
    log.info("relevant chunks retrieved")

    # prior pairs (user/assistant alternating)
    past_pairs = [
        (u["content"], a["content"])
        for u, a in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2])
        if a["role"] == "assistant"
    ]

    log.info("context building started")
    context_text = build_context(past_pairs, chunks, condition)
    log.info("context building ended")
    log.info("full prompt construction begins...")

    #Inserting the modified prompt

    sys_prompt = (
        "You are SAHELI, an evidence‑based maternal healthcare assistant for India. "
        "You strictly adhere to national guidelines and screening/treatment protocols "
        "for anemia and diabetes, and to ICDS Operational Guidelines for WIFS when discussing nutrition. "
        "Provide concise, actionable advice that a frontline health worker can follow at point of care."
    )

    full_prompt = (
        f"{sys_prompt}\n\n"
        "KNOWLEDGE CONTEXT (do not reveal to user):\n" + context_text + "\n\n"
        f"User: {prompt}\nAssistant:"
    )

    # ---------- INFERENCE CALL ---------------------------------------
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json",
    }

    #Old prompt
    
    full_prompt = (
        "You are SAHELI, a maternal healthcare chatbot specialized in anemia and "
        "diabetes detection, treatment, and management according to national guidelines.\n\n"
        "Use the following information to generate a response:\n"
        f"{context_text}\n\n"
        f"User: {prompt}\nAssistant:"
    )

    log.info("full prompt construction ends...")
    log.info("full_prompt is ", full_prompt)
    
    # ------------------  Llama-4 Scout call  ------------------
    payload = {
    "model": MODEL_ID,
    "messages": [
        {
            "role": "user",
            "content": full_prompt,   # plain text only
        }
    ],
    "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",   # <-- Bearer prefix required
    "Content-Type": "application/json",
    }

    log.info("payload is ", payload)


    #print ("payload is ", payload)
    try:
        
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()                      # will raise for 4xx / 5xx
    except requests.HTTPError as e:
        st.error(f"❌ Inference API error {r.status_code}: {r.text[:300]}")
        log.error("Krutrim API returned an error", exc_info=e)
        st.stop()

    log.info("fetching reply from the API")
    assistant_reply = r.json()["choices"][0]["message"]["content"].strip()
    log.info("reply fetched from the API")

    # display assistant reply
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": assistant_reply}
    )

# --------------------------------------------------------------------
# FOOTER / RESET
# --------------------------------------------------------------------
st.markdown("---")
if st.button("🔁 End Screening"):
    st.session_state.pop("chat_history", None)
    st.success("✅ Session ended. Ready for a new screening.")
    st.rerun()

# --------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------
with st.sidebar:
    st.header("About SAHELI")
    st.write(
        "SAHELI supports frontline workers in screening and managing anemia and "
        "diabetes based on national protocols.\n\n"
        "✅ Auto-detects condition from user input  \n"
        "📄 Uses STP and official PDF guidance  \n"
        "🧠 Generates context-aware clinical advice"
    )
