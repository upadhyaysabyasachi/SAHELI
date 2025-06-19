import pdfplumber
import pandas as pd
import torch
import pickle
from sentence_transformers import SentenceTransformer, util

# -------------------- Load PDF Chunks --------------------
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
anemia_chunks = anemia_chunks + load_chunks("ICMR_Anemia.pdf")

# -------------------- Load Excel Steps --------------------
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

steps_context = load_steps()

# -------------------- Create Embeddings --------------------
print("Loading BioBERT model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Encoding chunks...")
condition_embeddings = {
    "anemia": embedder.encode(anemia_chunks[:10], convert_to_tensor=True),
    "diabetes": embedder.encode(diabetes_chunks[:10], convert_to_tensor=True),
    "nutrition": embedder.encode(nutrition_chunks[:10], convert_to_tensor=True)
}

print("Encoding stepwise screening protocols...")
steps_embeddings = {
    key: embedder.encode(value, convert_to_tensor=True)
    for key, value in steps_context.items()
}

# -------------------- Save to Pickle --------------------
print("Saving to pickle...")
with open("embedding_minilm.pkl", "wb") as f:
    pickle.dump({
        "embedder_model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "condition_embeddings": condition_embeddings,
        "steps_context": steps_context,
        "steps_embeddings": steps_embeddings
    }, f)

print("✅ Done. Embeddings saved in embedding_data.pkl")
