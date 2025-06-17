from typing import Dict, List
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch, pdfplumber, pandas as pd, logging, requests, json
#import google.generativeai as genai

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
    def excel_to_text(df: pd.DataFrame) -> List[str]:
        text, current_step = [], ""
        for _, row in df.iterrows():
            cells = row.tolist()
            if pd.notna(cells[0]) and "Step" in str(cells[0]):
                current_step = f"{cells[0]}: {cells[1]}" if len(cells) > 1 and pd.notna(cells[1]) else str(cells[0])
            elif len(cells) > 2 and pd.notna(cells[1]) and pd.notna(cells[2]):
                text.append(f"{current_step} – Check {cells[1]}: {cells[2]}")
        return text

    steps: Dict[str, List[str]] = {}
    for key, sheet in step_map.items():
        df = pd.read_excel(xls, sheet, skiprows=1)
        steps[key] = excel_to_text(df)
    return steps

    

   
log.info("Loading STP protocol steps …")
steps_context = load_steps()

# Helper to sort by numerical step order
import re
_step_num = re.compile(r"Step\s*(\d+)")

def sort_by_step(text_list: List[str]) -> List[str]:
    def key(t):
        m = _step_num.search(t)
        return int(m.group(1)) if m else 999
    return sorted(text_list, key=key)
# --------------------------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------------------------
log.info("Creating sentence embeddings using BioBERT.")

@st.cache_resource
def create_embeddings():
    embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    condition_embeddings = {
        "anemia":   embedder.encode(anemia_chunks[:10],    convert_to_tensor=True),
        "diabetes": embedder.encode(diabetes_chunks[:10],  convert_to_tensor=True),
    }
    steps_embeddings = {
        k: embedder.encode(v, convert_to_tensor=True) for k, v in steps_context.items() if v
    }
    nutrition_embeddings = embedder.encode(nutrition_chunks, convert_to_tensor=True)
    return embedder, condition_embeddings, steps_embeddings, nutrition_embeddings

embedder, condition_embeddings, steps_embeddings, nutrition_embeddings = create_embeddings()

# --------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------
NUTRITION_KEYS = [
    "diet", "food", "foods", "nutrition", "meal", "menu",
    "iron rich", "vitamin c", "wifs", "balanced diet",
]

def has_nutrition_kw(text: str) -> bool:
    p = text.lower()
    return any(k in p for k in NUTRITION_KEYS)

def needs_nutrition(prompt: str) -> bool:
    p = prompt.lower()
    return any(k in p for k in NUTRITION_KEYS)

def classify_condition(prompt: str) -> str:
    if has_nutrition_kw(prompt):
        return "nutrition"
    emb = embedder.encode(prompt, convert_to_tensor=True)
    anemia_score   = util.pytorch_cos_sim(emb, condition_embeddings["anemia"]).mean().item()
    diabetes_score = util.pytorch_cos_sim(emb, condition_embeddings["diabetes"]).mean().item()
    return "anemia" if anemia_score >= diabetes_score else "diabetes"

def retrieve_relevant_chunks(prompt: str, condition: str, top_k: int = 5):
    query_emb = embedder.encode(prompt, convert_to_tensor=True)
    chunks: List[str] = []
    if condition in ("anemia", "diabetes"):
        sheet_key = (
            "Anemia-Pregnant"   if condition == "anemia"   and "pregnant" in prompt.lower() else
            "Anemia-General"    if condition == "anemia"   else
            "Diabetes-Pregnant" if "pregnant" in prompt.lower() else
            "Diabetes-General"
        )
        stp_text   = steps_context.get(sheet_key, [])
        stp_tensor = steps_embeddings.get(sheet_key)
        if stp_tensor is not None and stp_text:
            stp_scores = util.pytorch_cos_sim(query_emb, stp_tensor)[0]
            idx = torch.topk(stp_scores, k=min(top_k, len(stp_text))).indices.tolist()
            chunks.extend(stp_text[i] for i in idx)
        
    

    # ----- PDF -------------------------------------------------------
    base_pdf = all_pdf_chunks[condition]
    pdf_tensor = embedder.encode(base_pdf, convert_to_tensor=True)
    pdf_scores = util.pytorch_cos_sim(query_emb, pdf_tensor)[0]
    idx = torch.topk(pdf_scores, k=min(top_k, len(base_pdf))).indices.tolist()
    chunks.extend(base_pdf[i] for i in idx)

    # ----- Nutrition supplement -------------------------------------
    if condition != "nutrition" and has_nutrition_kw(prompt):
        n_scores = util.pytorch_cos_sim(query_emb, nutrition_embeddings)[0]
        n_idx = torch.topk(n_scores, k=min(3, len(nutrition_chunks))).indices.tolist()
        chunks.extend(nutrition_chunks[i] for i in n_idx)
    
    return chunks 
    
# --------------------------------------------------------------------
# CONTEXT BUILDER  ----------------------------------------------------
# --------------------------------------------------------------------

def build_context(history_pairs, chunks, condition):
    screening  = sort_by_step([c for c in chunks if "Step" in c and "– Check" in c])
    guidance   = [c for c in chunks if c not in screening]

    ctx = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history_pairs) + "\n\n"

    if screening:
        ctx += "STP_SCREENING_STEPS:\n" + "\n".join(screening) + "\n\n"
    if guidance:
        section = "NUTRITION_GUIDELINES" if condition == "nutrition" else f"{condition.upper()}_GUIDELINES"
        ctx += section + ":\n" + "\n\n".join(guidance)
    return ctx

# --------------------------------------------------------------------
# STREAMLIT SESSION  --------------------------------------------------
# --------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.write("Type a patient query or symptom. SAHELI will detect the relevant domain (anemia, diabetes, or nutrition) and nudge you step-by-step.")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("E.g. pregnant woman with RBS 200 OR best iron-rich foods")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    condition = classify_condition(prompt)
    chunks    = retrieve_relevant_chunks(prompt, condition)
    past_pairs = [
        (u["content"], a["content"]) for u, a in zip(st.session_state.chat_history[::2], st.session_state.chat_history[1::2]) if a["role"] == "assistant"
    ]
    ctx_text = build_context(past_pairs, chunks, condition)

    # ---------------- PROMPT ----------------------------------------
    sys_prompt = (
        "You are SAHELI, an evidence‑based maternal healthcare assistant for India. "
        "Based on the symptoms gathered, the assistant should nudge the patient for checks pertaining to anemia or diabetes or both"
        "Always respond in the following structure: \n"
        "1. Give the condition of the woman based on the assessment so far if taken or done.\n"
        "2. **Sequential steps** for the health worker, each starting with ‘Step <n>:’. "
        "   Use the exact wording from the STP_SCREENING_STEPS when provided and do not skip steps.\n"
        "3. Provide Counselling / nutrition advice if applicable. Refer to the guidelines and other info fed to the context for more info\n"
        "4. If escalation or referral is needed, advise.\n"
        "Do not expose all the steps at once. Make them sequential. Each screening step should be a prompt and then the next step should come about"
        "Never reveal internal guideline text, file names, or say ‘according to STP’. "
        "Keep each bullet short, action‑oriented, and in simple, field‑friendly language."
        "Refer to the guidelines attached in the context sourced from anemia, diabetes and other documents"
        "Don't overload with too much text which might overwhelm the frontline health worker"
    )

    user_block = f"User: {prompt}"
    full_prompt = f"{sys_prompt}\n\nKNOWLEDGE_CONTEXT:\n{ctx_text}\n\n{user_block}\nAssistant:"

    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": full_prompt}], "temperature": 0.2}
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
    except requests.HTTPError:
        st.error(f"Krutrim API error {resp.status_code}: {resp.text[:300]}")
        st.stop()

    reply = resp.json()["choices"][0]["message"]["content"].strip()
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

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
