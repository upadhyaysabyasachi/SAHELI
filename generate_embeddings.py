#!/usr/bin/env python3
"""
generate_embeddings.py
----------------------
Create embeddings.pkl containing:
  • condition_chunks / condition_embeddings
  • nutrition_chunks  / nutrition_embeddings
  • steps_context     / steps_embeddings
  • model_name        (string you passed to SentenceTransformer)
"""
import pickle, argparse, pdfplumber, pandas as pd, torch
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------- helpers --------------------------------------------------
def pdf_to_paragraphs(path: Path) -> List[str]:
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            if txt := page.extract_text():
                out.append(" ".join(line.strip() for line in txt.split("\n") if line.strip()))
    return out

def excel_to_steps(path: Path) -> Dict[str, List[str]]:
    xls = pd.ExcelFile(path)
    step_map = {
        "Anemia-Pregnant":   xls.sheet_names[0],
        "Anemia-General":    xls.sheet_names[1],
        "Diabetes-Pregnant": xls.sheet_names[2],
        "Diabetes-General":  xls.sheet_names[3],
    }
    def _proc(df):
        cur, buff = "", []
        for _, row in df.iterrows():
            row = row.tolist()
            if pd.notna(row[0]) and "Step" in str(row[0]):
                cur = f"{row[0]}: {row[1]}" if len(row)>1 and pd.notna(row[1]) else row[0]
            elif len(row)>2 and pd.notna(row[1]) and pd.notna(row[2]):
                buff.append(f"{cur} - Check {row[1]}: {row[2]}")
        return buff
    return {lbl: _proc(pd.read_excel(xls, sh, skiprows=1)) for lbl, sh in step_map.items()}

# ---------- main -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                    help="Sentence-Transformer model ID")
    ap.add_argument("--out", "-o", default="embeddings.pkl", help="Output pickle")
    args = ap.parse_args()

    # document paths
    anemia_pdf    = Path("data.pdf")
    diabetes_pdf  = Path("diabetes.pdf")
    nutrition_pdf = Path("icds_operational_guidelines_for_wifs.pdf")
    stp_excel     = Path("STP_v2.xlsx")

    print("▶ Loading documents …")
    anemia_chunks    = pdf_to_paragraphs(anemia_pdf)
    diabetes_chunks  = pdf_to_paragraphs(diabetes_pdf)
    nutrition_chunks = pdf_to_paragraphs(nutrition_pdf)
    steps_context    = excel_to_steps(stp_excel)

    print(f"▶ Encoding with {args.model} …")
    model = SentenceTransformer(args.model)

    condition_embeddings = {
        "anemia":   model.encode(anemia_chunks,   convert_to_tensor=True, show_progress_bar=True),
        "diabetes": model.encode(diabetes_chunks, convert_to_tensor=True, show_progress_bar=True),
    }
    steps_embeddings = {
        k: model.encode(v, convert_to_tensor=True)
        for k, v in tqdm(steps_context.items(), desc="STP sheets")
    }
    nutrition_embeddings = model.encode(nutrition_chunks, convert_to_tensor=True, show_progress_bar=True)

    blob = dict(
        condition_chunks     = {"anemia": anemia_chunks, "diabetes": diabetes_chunks},
        condition_embeddings = {k: v.cpu() for k, v in condition_embeddings.items()},
        nutrition_chunks     = nutrition_chunks,
        nutrition_embeddings = nutrition_embeddings.cpu(),
        steps_context        = steps_context,
        steps_embeddings     = {k: v.cpu() for k, v in steps_embeddings.items()},
        model_name           = args.model,                       # <-- changed line
        dim                  = model.get_sentence_embedding_dimension(),
    )
    with open(args.out, "wb") as f:
        pickle.dump(blob, f)
    print(f"✅ Saved all embeddings → {args.out}")

if __name__ == "__main__":
    main()
