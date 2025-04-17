import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import pdfplumber
import pandas as pd

# -------------------- ENVIRONMENT SETUP --------------------
# Configure the Gemini API with the secret key stored in Streamlit's secrets manager for secure AI model access
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# -------------------- PAGE SETUP --------------------
# Configure the Streamlit page with a wide layout and set the title for the maternal healthcare assistant interface
st.set_page_config(page_title="SAHELI Assistant", layout="wide")
st.title("🤖 SAHELI: Maternal Healthcare Assistant")

# -------------------- MODEL LOADING --------------------
# Cache the Gemini model to prevent reloading on each interaction, improving response time
@st.cache_resource
def load_gemini_model():
    return genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

model = load_gemini_model()

# -------------------- LOAD PDF --------------------
# Extract and process PDF content into manageable text chunks for retrieval and context building
@st.cache_data
def load_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Clean and format the extracted text by removing whitespace and joining lines into coherent paragraphs
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                paragraph = ' '.join(lines)
                chunks.append(paragraph)
    return chunks

chunks = load_chunks('data.pdf')

# -------------------- LOAD EXCEL --------------------
# Load structured clinical protocols from Excel files with different sheets for various conditions and patient states
@st.cache_data
def load_all_excel_steps(excel_path='STP.xlsx'):
    try:
        xls = pd.ExcelFile(excel_path)
        steps = {}
        for sheet in xls.sheet_names:
            # Process each sheet containing different clinical pathways (anemia, diabetes, pregnancy variations)
            df = pd.read_excel(xls, sheet, skiprows=1)
            steps[sheet] = excel_to_text(df)
        return steps
    except Exception as e:
        st.warning(f"Could not load Excel file: {e}")
        return {}

# Convert tabular Excel data into sequential, readable clinical protocol steps for the AI assistant
def excel_to_text(df):
    steps_text = []
    current_step = ""
    for _, row in df.iterrows():
        if pd.notna(row[0]) and isinstance(row[0], str) and 'Step' in row[0]:
            # Identify and format step headers from the Excel structure
            current_step = row[0] + ": " + str(row[1]) if pd.notna(row[1]) else row[0]
        elif pd.notna(row[1]) and pd.notna(row[2]):
            # Extract specific examination points and their clinical significance
            steps_text.append(f"{current_step} - Check {row[1]}: {row[2]}")
    return steps_text

all_steps = load_all_excel_steps()

# -------------------- EMBEDDINGS --------------------
# Create vector embeddings for both PDF content and structured protocol steps to enable semantic search
@st.cache_resource
def create_embeddings(text_chunks, all_steps):
    # Use a biomedical-specific BERT model that understands medical terminology and concepts
    embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
    # Generate embeddings for general healthcare guidelines from PDF
    pdf_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    # Create separate embeddings for each clinical pathway to allow condition-specific retrieval
    step_embeddings = {
        k: embedder.encode(v, convert_to_tensor=True)
        for k, v in all_steps.items() if v
    }
    return embedder, pdf_embeddings, step_embeddings

embedder, corpus_embeddings, step_embeddings = create_embeddings(chunks, all_steps)

# -------------------- CONDITION SELECTION --------------------
# Manage the application state for selecting which clinical condition to screen for
if 'pathway_selected' not in st.session_state:
    st.session_state.pathway_selected = False
if 'selected_condition' not in st.session_state:
    st.session_state.selected_condition = ""

# Display initial condition selection interface if no pathway has been chosen yet
if not st.session_state.pathway_selected:
    condition = st.radio("Which condition are you screening for?", ["Anemia", "Diabetes"])
    if st.button("Start Screening"):
        st.session_state.selected_condition = condition
        st.session_state.pathway_selected = True
        st.experimental_rerun()
else:
    # Once a condition is selected, initiate the specialized clinical assessment workflow
    selected_condition = st.session_state.selected_condition

    # Define condition-specific assessment questions based on standard clinical protocols for each condition
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

    # Initialize session state variables to track progress through the assessment workflow
    if 'step_index' not in st.session_state:
        st.session_state.step_index = 0
    if 'step_responses' not in st.session_state:
        st.session_state.step_responses = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Get the appropriate sequence of screening questions for the selected health condition
    stepwise_questions = stepwise_questions_map[selected_condition]

    # Display a summary of all previously collected patient information for context
    st.subheader("📝 Collected Observations")
    for step in st.session_state.step_responses:
        st.markdown(f"- {step}")

    # Display current assessment question if the screening process is still ongoing
    if st.session_state.step_index < len(stepwise_questions):
        current_question = stepwise_questions[st.session_state.step_index]
        user_input = st.text_input(current_question, key=f"step_{st.session_state.step_index}")
        if user_input:
            # Store the healthcare worker's response and advance to the next assessment step
            st.session_state.step_responses.append(f"{current_question} — Response: {user_input}")
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.step_index += 1
            st.experimental_rerun()

    else:
        # After completing all assessment steps, prepare for diagnosis generation
        # Compile all collected patient information into a comprehensive summary
        user_summary = "\n".join(st.session_state.step_responses)
        prompt = f"Proceed with {selected_condition.lower()} diagnosis based on observations."

        # Select the appropriate clinical protocol based on condition and pregnancy status
        if selected_condition == "Anemia":
            steps_key = "Sheet1" if "pregnant" in user_summary.lower() else "Sheet2"
        else:
            steps_key = "Diabetes-Pregnant" if "pregnant" in user_summary.lower() else "Diabetes-NonPregnant"

        # Retrieve the relevant clinical protocol steps for the specific patient situation
        relevant_chunks = all_steps.get(steps_key, [])
        embeddings = step_embeddings.get(steps_key)

        # Use semantic search to identify the most relevant clinical protocol steps for this case
        if embeddings:
            query_embedding = embedder.encode(user_summary, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
            top_indices = torch.topk(scores, k=min(5, len(relevant_chunks))).indices.tolist()
            retrieved_chunks = [relevant_chunks[i] for i in top_indices]
        else:
            retrieved_chunks = []

        # Build the context for the AI by combining protocol steps and guidelines
        context_chunks = retrieved_chunks
        if selected_condition == "Anemia":
            context_chunks += chunks  # Add additional anemia-specific guidelines from the PDF

        context = "\n".join(context_chunks)

        # Construct a detailed prompt for the AI with patient data and relevant clinical guidelines
        full_prompt = f"""You are SAHELI, a maternal healthcare chatbot specialized in {selected_condition.lower()} detection, screening, and follow-up based on national health protocols.

Here are the observations:
{user_summary}

Relevant Guidelines:
{context}

User: {prompt}
Assistant:"""

        # Generate a clinical assessment and recommendations using the Gemini AI model
        response = model.generate_content(full_prompt)
        answer = response.text.strip()

        # Display the AI-generated clinical guidance to the healthcare worker
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # -------------------- FOLLOW-UP SUPPORT --------------------
        # Provide an interface for healthcare workers to ask follow-up questions about the diagnosis
        st.subheader("💬 Continue the Conversation")
        col1, col2 = st.columns([3, 1])
        with col1:
            followup_prompt = st.chat_input("Ask a follow-up question or type 'end' to finish")
        with col2:
            # Offer a button to explicitly end the screening session and reset the application
            if st.button("🔁 End Screening"):
                followup_prompt = "end"

        # Process follow-up questions or handle the end of the screening session
        if followup_prompt:
            if followup_prompt.strip().lower() == "end":
                # Reset all session variables to prepare for a new patient assessment
                for key in ["pathway_selected", "selected_condition", "step_index", "step_responses", "chat_history"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Screening ended. Ready for the next patient.")
                st.experimental_rerun()
            else:
                # Handle follow-up questions by storing them in chat history
                st.session_state.chat_history.append({"role": "user", "content": followup_prompt})

                # Compile the entire conversation history to maintain context for the AI
                history_text = "\n".join([
                    f"User: {m['content']}" if m['role'] == 'user' else f"Assistant: {m['content']}"
                    for m in st.session_state.chat_history
                ])
                continued_prompt = f"""You are SAHELI, continuing a medical support conversation for {selected_condition.lower()} with a healthcare worker.

Conversation so far:
{history_text}

Respond to the latest user query with appropriate clinical guidance, referring to earlier context and official protocol if needed."""

                # Generate a contextually aware response to the follow-up question
                continued_response = model.generate_content(continued_prompt)
                continued_answer = continued_response.text.strip()

                # Display the AI's response to the follow-up question
                with st.chat_message("assistant"):
                    st.markdown(continued_answer)
                st.session_state.chat_history.append({"role": "assistant", "content": continued_answer})

        # Display contextual information about the app and selected condition in the sidebar
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
