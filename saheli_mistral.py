import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import pandas as pd
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
from transformers import BitsAndBytesConfig
import os

# -------------------- CONFIGURATION --------------------
login(token=st.secrets["HUGGINGFACE_KEY"])
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

log = logging.getLogger("SAHELI")

def _supports_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0

def _import_accelerate():
    try:
        import accelerate  # noqa: F401
        return True
    except ImportError:
        return False

# -------------------- CACHED LOADER --------------------
@st.cache_resource(show_spinner="Loading Mistral 7B…")
def load_model():
    """
    Returns a text-generation pipeline powered by Mistral 7B if possible;
    otherwise falls back to Gemini (already in your project).
    """
    if not _supports_cuda() and torch.cuda.is_available():
        # edge case: CUDA present but not initialised correctly
        log.warning("CUDA detected but unusable – falling back to CPU.")
    
    

    HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
    HF_TOKEN = st.secrets.get("HUGGINGFACE_KEY", None)

    if (_supports_cuda() or os.getenv("FORCE_MISTRAL_CPU")) and _import_accelerate():
        log.info("Initialising Mistral 7B (%s)…", "GPU" if _supports_cuda() else "CPU/4-bit")

        quant_cfg = None
        if not _supports_cuda():
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, use_fast=True, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL,
            device_map="auto",
            token=HF_TOKEN,
            quantization_config=quant_cfg,
            trust_remote_code=True,
        )
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if _supports_cuda() else -1,
            max_new_tokens=512,
            temperature=0.3,
        )

#@st.cache_resource
#def load_model():
#    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#    tokenizer = AutoTokenizer.from_pretrained(model_name)
#    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
#    return pipeline("text-generation", model=model, tokenizer=tokenizer)

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
nutrition_chunks = load_chunks("icds_operational_guidelines_for_wifs.pdf")

all_pdf_chunks = {"anemia": anemia_chunks, "diabetes": diabetes_chunks, "nutrition": nutrition_chunks}

# -------------------- LOAD EXCEL STP --------------------
# (use your existing Excel-loading logic here)

# -------------------- EMBEDDINGS --------------------
@st.cache_resource
def create_embeddings():
    embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    condition_embeddings = {
        "anemia": embedder.encode(anemia_chunks[:10], convert_to_tensor=True),
        "diabetes": embedder.encode(diabetes_chunks[:10], convert_to_tensor=True)
    }
    nutrition_embeds = embedder.encode(nutrition_chunks, convert_to_tensor=True)
    return embedder, condition_embeddings, nutrition_embeds

embedder, condition_embeddings, nutrition_embeds = create_embeddings()

# -------------------- RETRIEVAL AND RESPONSE --------------------
def retrieve_relevant_chunks(prompt, selected_condition, top_k=3):
    query_embedding = embedder.encode(prompt, convert_to_tensor=True)

    pdf_chunks = all_pdf_chunks[selected_condition]
    pdf_tensor = embedder.encode(pdf_chunks, convert_to_tensor=True)
    pdf_scores = util.pytorch_cos_sim(query_embedding, pdf_tensor)[0]
    pdf_idx = torch.topk(pdf_scores, k=min(top_k, len(pdf_chunks))).indices.tolist()
    pdf_hits = [pdf_chunks[i] for i in pdf_idx]

    return pdf_hits

# -------------------- SESSION INIT --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- INPUT HANDLER --------------------
if prompt := st.chat_input("E.g. pregnant woman with RBS 200"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    anemia_score = util.pytorch_cos_sim(embedder.encode(prompt), condition_embeddings["anemia"]).mean()
    diabetes_score = util.pytorch_cos_sim(embedder.encode(prompt), condition_embeddings["diabetes"]).mean()
    condition = "anemia" if anemia_score >= diabetes_score else "diabetes"

    chunks = retrieve_relevant_chunks(prompt, condition)

    context = "\n".join(chunks)

    full_prompt = f"""You are SAHELI, a maternal healthcare assistant specialized in {condition}. Answer briefly using these guidelines:\n{context}\nUser: {prompt}\nAssistant:"""

    response = model(full_prompt, max_length=500, do_sample=True, temperature=0.7)[0]["generated_text"]
    answer = response.split("Assistant:")[-1].strip()

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.markdown("---")
if st.button("🔁 End Screening"):
    st.session_state.chat_history = []
    st.success("✅ Session ended. Ready for a new screening.")
    st.rerun()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("About SAHELI")
    st.write("""
SAHELI supports frontline workers in screening and managing anemia and diabetes based on national protocols.

✅ Auto-detects condition from user input  
📄 Uses official PDF guidelines  
🧠 Generates context-aware clinical advice  
""")
