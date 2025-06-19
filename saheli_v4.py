import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pdfplumber
import pandas as pd
import pickle
import requests
import logging

# -------------------- ENVIRONMENT SETUP --------------------
API_URL = "https://cloud.olakrutrim.com/v1/chat/completions"
MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct"
BEARER_TOKEN = st.secrets["BEARER_TOKEN"]

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

@st.cache_resource
def load_embedding_data(path="embedding_minilm.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)

    embedder = SentenceTransformer(data["embedder_model_id"])
    return embedder, data["condition_embeddings"], data["steps_context"], data["steps_embeddings"]

embedder, condition_embeddings, steps_context, steps_embeddings = load_embedding_data()

# -------------------- PATHWAY SELECTION --------------------
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

    st.subheader("📜 Collected Observations")
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

        relevant_chunks = steps_context.get(steps_key, [])
        embeddings = steps_embeddings.get(steps_key)

        if embeddings:
            query_embedding = embedder.encode(user_summary, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_indices = torch.topk(scores, k=min(5, len(relevant_chunks))).indices.tolist()
            retrieved_chunks = [relevant_chunks[i] for i in top_indices]
        else:
            retrieved_chunks = []

        context_chunks = retrieved_chunks
        context = "\n".join(context_chunks)

        full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in {selected_condition.lower()} detection, screening, and follow-up based on national health protocols.

Here are the observations:
{user_summary}

Relevant Guidelines:
{context}

User: {prompt}
Assistant:"""

        # ------------------  Krutrim API call ------------------
        payload = {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.2
        }

        headers = {
            "Authorization": f"Bearer {BEARER_TOKEN}",
            "Content-Type": "application/json"
        }

        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            assistant_reply = r.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            st.error(f"❌ Inference API error {r.status_code}: {r.text[:300]}")
            assistant_reply = "An error occurred while generating the response."

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

        # Follow-up support
        st.subheader("💬 Continue the Conversation")
        followup_prompt = st.chat_input("Ask a follow-up question or say 'end' to finish")

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

                payload = {
                    "model": MODEL_ID,
                    "messages": [
                        {
                            "role": "user",
                            "content": continued_prompt
                        }
                    ],
                    "temperature": 0.2
                }

                try:
                    r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                    r.raise_for_status()
                    continued_answer = r.json()["choices"][0]["message"]["content"].strip()
                except requests.HTTPError as e:
                    st.error(f"❌ Follow-up API error {r.status_code}: {r.text[:300]}")
                    continued_answer = "An error occurred while generating the follow-up response."

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
