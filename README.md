# SAHELI - Maternal Healthcare Chatbot for Anemia Detection

SAHELI (Smart Assistant for Healthcare Education and Localized Information) is an AI-powered chatbot specialized in anemia detection, management, and treatment according to the Anemia Mukt Bharat (AMB) guidelines.

## Features

- Step-by-step anemia screening protocols
- Clinical decision support
- Treatment guidelines
- Follow-up recommendations

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Google Gemini API key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/SAHELI.git
   cd SAHELI
   ```

2. Create a virtual environment and activate it:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### API Key Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file and replace `your_gemini_api_key_here` with your actual Google Gemini API key.

### Running the Application

Start the Streamlit application:
```bash
streamlit run saheli.py
```

## Important Notes About API Keys

- Never commit your actual API keys to version control
- Always use environment variables or a `.env` file (which is excluded from git)
- Make sure `.env` is included in your `.gitignore` file
- Share the `.env.example` file with your team to show which environment variables are needed

## Switching to a Virtual Environment

To switch to the virtual environment from the terminal:

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

To deactivate the virtual environment:
```bash
deactivate
```

## Data Sources

The chatbot uses two main sources of data:
1. PDF document with Anemia Mukt Bharat guidelines
2. Excel file with structured screening protocol

# Main file - saheli.py
# Embedding/sentence transformer model - BERT
# base model - gemini-pro-experimental-2.5

