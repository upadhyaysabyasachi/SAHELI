import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import pandas as pd
import hashlib
import json
import os
from datetime import datetime

# -------------------- SETUP --------------------
# Configure Gemini API with secure key from Streamlit secrets and set up the application's UI with a wide layout for better readability
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

# Check if a session just ended and handle the reset flow
if st.session_state.get('show_session_ended_message', False):
    st.success("✅ All data saved.")
    st.info("Session ended. Click below to start a new screening.")

    if st.button("✨ Start New Screening Session"):
        # Clear all keys related to the previous screening session
        keys_to_clear = [
            k for k in st.session_state.keys()
            if k.endswith('_responses') or k.endswith('_followup') or k == 'interaction_log' or k == 'show_session_ended_message'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Rerun to get a completely fresh start
        st.experimental_rerun()
    # Stop execution here; don't render the rest of the app until 'Start New' is clicked
    st.stop()

# -------------------- PATIENT SETUP --------------------
# Generate a unique patient identifier by hashing the combination of name and phone number for privacy and consistent identification
def generate_patient_id(name, phone):
    base = f"{name.strip().lower()}_{phone.strip()}"
    return hashlib.md5(base.encode()).hexdigest()[:8]

# Load previous screening records and interaction history for a returning patient from their JSON file
def load_patient_history(patient_id):
    path = f"patients/{patient_id}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"interactions": []}

# Persist patient data including screening results and recommendations to enable longitudinal care tracking
def save_patient_history(patient_id, data):
    os.makedirs("patients", exist_ok=True)
    with open(f"patients/{patient_id}.json", "w") as f:
        json.dump(data, f, indent=2)

# -------------------- LOAD MODELS & DATA --------------------
# Cache the Gemini AI model in memory to prevent reloading between user interactions, improving application responsiveness
@st.cache_resource
def load_model():
    return genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

# Extract and process clinical guidelines from PDF, converting them into manageable text chunks for retrieval and context building
@st.cache_data
def load_pdf_chunks(pdf_paths=None):
    """
    Load and process multiple PDFs for different conditions
    
    Args:
        pdf_paths: Dictionary mapping condition names to PDF paths
                  (defaults to {'Anemia': 'data.pdf', 'Diabetes': 'diabetes.pdf'})
    
    Returns:
        Dictionary mapping condition names to their processed text chunks
    """
    if pdf_paths is None:
        pdf_paths = {
            'Anemia': 'data.pdf',
            'Diabetes': 'diabetes.pdf'
        }
    
    condition_chunks = {}
    
    for condition, path in pdf_paths.items():
        chunks = []
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        chunks.append(" ".join([line.strip() for line in text.split('\n') if line.strip()]))
            condition_chunks[condition] = chunks
        except Exception as e:
            st.warning(f"Error loading PDF for {condition}: {e}")
            condition_chunks[condition] = []
    
    return condition_chunks

# Load structured clinical protocols from Excel, where each sheet represents a different condition or patient demographic variant
@st.cache_data
def load_protocol_steps(excel_path='STP_v2.xlsx'):
    xls = pd.ExcelFile(excel_path)
    steps = {}
    for sheet in xls.sheet_names:
        # Parse each Excel sheet containing protocol steps, skipping header row to access actual content
        df = pd.read_excel(xls, sheet, skiprows=1)
        steps[sheet] = excel_to_text(df)
    return steps

# Transform structured tabular clinical data into sequential, readable protocol steps for the AI to process and reference
def excel_to_text(df):
    text = []
    current_step = ""
    
    # First, check if the DataFrame has enough columns
    num_columns = len(df.columns)
    
    for _, row in df.iterrows():
        # Handle the case when row has at least 1 column
        if num_columns > 0 and pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in str(row[0]):
            # For Step headers, we need at least the step label
            if num_columns > 1 and pd.notna(row[1]):
                current_step = f"{row[0]}: {row[1]}"
            else:
                current_step = f"{row[0]}"
        # Handle the case when row has at least 3 columns for detailed steps
        elif num_columns > 2 and pd.notna(row[1]) and pd.notna(row[2]):
            # Extract specific clinical assessment points and their diagnostic significance
            text.append(f"{current_step} - Check {row[1]}: {row[2]}")
        # Handle the case when row has only 2 columns
        elif num_columns > 1 and pd.notna(row[0]) and pd.notna(row[1]) and current_step:
            # Use a simpler format when there are only 2 columns
            text.append(f"{current_step} - {row[0]}: {row[1]}")
    
    return text

# Create vector embeddings for semantic search, using a biomedical-specialized model for better healthcare concept understanding
@st.cache_resource
def create_embeddings(chunks_dict, steps_dict):
    # Use domain-specific BERT model trained on medical literature
    embedder = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    # Generate embeddings for each condition's guidelines
    pdf_embeds = {}
    for condition, chunks in chunks_dict.items():
        if chunks:  # Skip empty chunks
            pdf_embeds[condition] = embedder.encode(chunks, convert_to_tensor=True)
    
    # Create embeddings for clinical pathways
    step_embeds = {k: embedder.encode(v, convert_to_tensor=True) for k, v in steps_dict.items() if v}
    
    return embedder, pdf_embeds, step_embeds

# -------------------- LOAD ALL --------------------
# Initialize all required components with condition-specific PDFs
model = load_model()

# Define paths to condition-specific PDFs
pdf_paths = {
    'Anemia': 'data.pdf',
    'Diabetes': 'diabetes.pdf'
    # Add more conditions and their PDF paths as needed
}

pdf_chunks_by_condition = load_pdf_chunks(pdf_paths)
stepwise_data = load_protocol_steps()

# Create embeddings for each condition's PDF content
condition_embeddings = {}
for condition, chunks in pdf_chunks_by_condition.items():
    if chunks:  # Only process non-empty chunks
        # embedder is not defined in this scope - it's only created later in create_embeddings()
        # This line should be removed since the embeddings are already created in create_embeddings()
        # condition_embeddings[condition] = embedder.encode(chunks, convert_to_tensor=True)
        pass  # Add this line to ensure proper indentation with an empty block

# Create embeddings for semantic search
embedder, pdf_embeddings, step_embeddings = create_embeddings(pdf_chunks_by_condition, stepwise_data)

# -------------------- PATIENT INPUT --------------------
# Collect and display patient identification information in the sidebar for easy reference during screening
st.sidebar.header("👩‍⚕️ Patient Information")
# Use a default value and check if it's empty to help with reset
name_input = st.sidebar.text_input("Patient Name", key="patient_name_input")
phone_input = st.sidebar.text_input("Phone Number", key="patient_phone_input")

# Assign to variables only if not empty
name = name_input if name_input else None
phone = phone_input if phone_input else None

# Halt application flow if essential patient identifiers are missing to ensure complete records
if not name or not phone:
    st.info("Please enter patient name and phone number in the sidebar to begin.")
    st.stop()

# Generate consistent patient ID and load any existing medical history for continuity of care
patient_id = generate_patient_id(name, phone)
st.sidebar.success(f"Patient ID: {patient_id}")
history = load_patient_history(patient_id)

# Display previous screening history in the sidebar to provide context for current assessment session
if history["interactions"]:
    st.sidebar.subheader("📜 Previous Screenings")
    for item in history["interactions"]:
        st.sidebar.markdown(f"- **{item['condition']}** on {item['timestamp']}")

# -------------------- CONDITION SELECTION --------------------
# Allow healthcare workers to select multiple conditions to screen for in a single session
st.subheader("📋 Select Conditions to Screen")
selected_conditions = st.multiselect("Choose one or more conditions", ["Anemia", "Diabetes"])

# Define standardized assessment questions for each condition based on clinical protocols
condition_questions = {
    "Anemia": [
        "Step 0: Is the woman pregnant or not?",
        "Step 1: Physical signs (pale eyelids, palms, tongue, or brittle nails)?",
        "Step 2: Symptoms (dizziness, tiredness, heartbeat, breathlessness)?",
        "Step 3: Hemoglobin value?",
        "Step 4: Any treatment started?"
    ],
    "Diabetes": [
        "Step 0: Is the person pregnant or not?",
        "Step 1: Fasting blood glucose level?",
        "Step 2: Postprandial blood glucose level?",
        "Step 3: Symptoms (thirst, urination, fatigue)?",
        "Step 4: Any meds/insulin currently?"
    ]
}

# Initialize session state to track current screening interactions for later persistence
if "interaction_log" not in st.session_state:
    st.session_state.interaction_log = []

# -------------------- SCREENING WORKFLOW --------------------
# Process each selected condition through its specific assessment workflow
for condition in selected_conditions:
    # Create a visual section for each condition being screened with distinct header
    st.markdown(f"## 🩺 {condition} Screening")
    questions = condition_questions[condition]
    responses_key = f"{condition}_responses"

    # Initialize condition-specific response tracking if first time encountering this condition
    if responses_key not in st.session_state:
        st.session_state[responses_key] = []

    responses = st.session_state[responses_key]

    # Display each clinical assessment question sequentially and collect healthcare worker's input
    for i, q in enumerate(questions):
        user_input = st.text_input(q, key=f"{condition}_q_{i}")
        if user_input and len(responses) == i:
            # Store response and refresh UI to progress to next question
            responses.append(f"{q} — {user_input}")
            st.experimental_rerun()

    # Once all assessment questions are answered, generate clinical guidance using AI
    if len(responses) == len(questions):
        # Combine all responses into a comprehensive patient summary
        summary = "\n".join(responses)
        
        # Select appropriate clinical protocol sheet based on condition and pregnancy status
        sheet_key = (
            "anemia-pregnant" if condition == "Anemia" and "pregnant" in summary.lower() else
            "anemia-general" if condition == "Anemia" else
            "diabetes-pregnant" if "pregnant" in summary.lower() else
            "diabetes-general"
        )

        # Retrieve protocol steps relevant to this specific condition and patient demographic
        relevant_steps = stepwise_data.get(sheet_key, [])
        step_embeds = step_embeddings.get(sheet_key)

        # Use semantic search to find most relevant protocol guidance for this specific patient case
        if step_embeds is not None and len(relevant_steps) > 0:
            query_vector = embedder.encode(summary, convert_to_tensor=True)
            try:
                sim_scores = util.pytorch_cos_sim(query_vector, step_embeds)[0]
                # If sim_scores is on GPU, move to CPU before using topk
                if sim_scores.device.type == 'cuda':
                    sim_scores = sim_scores.cpu()
                
                # Ensure the number of elements to select does not exceed the tensor size
                k_value = min(5, len(relevant_steps))
                if k_value > 0:
                    top_k = torch.topk(sim_scores, k=k_value).indices.tolist()
                    retrieved = [relevant_steps[i] for i in top_k]
                else:
                    retrieved = []
            except Exception as e:
                st.warning(f"Semantic search error: {e}")
                retrieved = relevant_steps[:5] if len(relevant_steps) >= 5 else relevant_steps
        else:
            retrieved = []

        # Build context using condition-specific PDF chunks
        condition_pdf_chunks = pdf_chunks_by_condition.get(condition, [])
        context = "\n".join(retrieved + condition_pdf_chunks)

        # --- Initial Recommendation Generation (Only if not already generated) ---
        recommendation_key = f"{condition}_recommendation"
        if recommendation_key not in st.session_state:
            # Construct detailed prompt combining patient data with clinical guidelines for AI assessment
            full_prompt = f"""You are SAHELI, a maternal healthcare assistant chatbot. Please assist in {condition.lower()} screening and follow-up.

You must:
1. Provide specific, actionable advice based strictly on the pdfs and excel sheets attached as a part of the context
2. Follow the structured screening protocol for detection of anemia
3. Recommend appropriate tests, treatments, and follow-ups based on evidence
4. Use simple, clear language appropriate for healthcare workers in rural India
5. Be concise but thorough in your explanations
6. Never invent symptoms, treatments, or recommendations not supported by the provided context
                
Patient Observations:
{summary}

Guidelines to refer:
{context}

Please give clinical advice strictly aligned with official protocols."""

            # Generate personalized clinical guidance based on patient data and medical protocols
            result = model.generate_content(full_prompt).text.strip()
            st.session_state[recommendation_key] = result

            # Log this initial screening interaction 
            st.session_state.interaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "condition": condition,
                "summary": summary,
                "response": result
            })
            
            # Initialize conversation history for follow-up chat 
            followup_key = f"{condition}_followup"
            st.session_state[followup_key] = [
                {"role": "system", "content": f"Previous screening summary for {condition}: {summary}"},
                {"role": "assistant", "content": result}
            ]
        
        # --- Display Recommendation and Persistent Follow-up Chat ---
        st.markdown("### 🧠 SAHELI's Recommendation")
        st.success(st.session_state[recommendation_key])

        followup_key = f"{condition}_followup"
        
        # Display the ongoing conversation history persistently
        st.subheader("Conversation History")
        if followup_key in st.session_state and st.session_state[followup_key]:
            for msg in st.session_state[followup_key]:
                if msg['role'] == 'system':
                    continue # Don't show system messages
                elif msg['role'] == 'user':
                    st.markdown(f"**You:** {msg['content']}")
                else: # assistant
                    st.markdown(f"**SAHELI:** {msg['content']}")
        else:
             st.info("No follow-up conversation yet.") # Initial state

        # Add a section for follow-up questions - always visible after recommendation
        st.markdown("### 💬 Ask Follow-up Questions")
        st.info("You can continue asking follow-up questions about this assessment.")
        
        followup_question = st.text_input(
            "Ask a question about this assessment:", 
            key=f"followup_{condition}_input" # Use a unique key for the input widget itself
        )
        
        # Process a NEW follow-up question if entered
        if followup_question:
            # Add user question to history
            st.session_state[followup_key].append({"role": "user", "content": followup_question})
            
            # Create context for follow-up using the current history
            followup_history = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'SAHELI'}: {msg['content']}" 
                for msg in st.session_state[followup_key] 
                if msg['role'] != 'system'
            ])
            
            # Construct prompt for follow-up
            followup_prompt = f"""You are SAHELI, continuing a conversation about {condition.lower()} screening and management.

Previous conversation:
{followup_history}

Use the context from the previous conversation to answer the user's latest question.
Focus on providing clinical guidance according to established protocols.
Be clear, specific, and helpful with your advice.

Guidelines to refer:
{context}"""

            # Generate response to follow-up question
            followup_response = model.generate_content(followup_prompt).text.strip()
            
            # Add assistant response to history
            st.session_state[followup_key].append({"role": "assistant", "content": followup_response})
            
            # Add this follow-up interaction to the main log
            st.session_state.interaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "condition": condition,
                "summary": f"Follow-up Q: {followup_question}",
                "response": followup_response
            })

            # Clear the text input field after processing by rerunning
            # Note: The conversation history will be redrawn correctly on rerun
            st.experimental_rerun() 
        
        # Add a button to clear this specific follow-up conversation 
        if st.button("Clear this Follow-up Conversation", key=f"clear_{condition}"):
            # Reset only the follow-up history, keep the initial recommendation
            st.session_state[followup_key] = [
                {"role": "system", "content": f"Previous screening summary for {condition}: {summary}"},
                {"role": "assistant", "content": st.session_state[recommendation_key]}
            ]
            st.experimental_rerun()

# -------------------- SAVE ALL --------------------
# Place this button strategically, maybe after the loop or where it makes sense in your UI flow
# Ensure it's only shown when there's something to save
if selected_conditions and st.session_state.get("interaction_log"):
    if st.button("💾 Save and End Session"):
        # Merge new screening data with existing patient history for comprehensive longitudinal record
        save_patient_history(patient_id, {
            "interactions": history["interactions"] + st.session_state.interaction_log
        })

        # Set the flag to trigger the 'session ended' state on the next run
        st.session_state.show_session_ended_message = True

        # Rerun immediately to show the message and 'Start New' button
        st.experimental_rerun()
