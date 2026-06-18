#Banking AI catalyst Toolkit

A suite of AI-powered banking operations tools built during AI catalyst learning journey phase 3 & 4

## What this does
- **Policy Q&A Bot** - Answer questions from banking policy documents using RAG (Retrieval Augmented Generation)
- **Transaction Risk Analyser** - Classifies transactions with structured JSON output
- **Compliant Classifier** - ROutes customer complaints using few-shot prompting
- **Guardrail COmpliance Assistant** - Banking operations assistant with compliance controls

## Tech Stack
- Python 3.11
- OpenAI API (GPT-4o-mini)
- LangChain
- FAISS vector database
- Streamlit (web interface)
- FastAPI (Rest API)

# Setup

1. Clone this repository 
    git clone https://github.com/KateCheong/ai-catalyst.git

2. Create virtual enviornment
    python -m venv ai-catalyst-env
    source ai-catalyst-env/bin/activate

3. Install dependencies
    pip install -r requirements.txt

4. Add your API key
    Create a .env file with:
    OPENAI_API_KEY=your-key-here

5. Run the policy Q&A bot
    streamlit run streamlit_app.py

## Important
All examples use sythentic data only.
Never use real customer data with public AI tools.
