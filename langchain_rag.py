#==================================================================
# Langchain_rag.py
# Professional RAG ststem built with Langchain
# This is production-grade banking AI
#==================================================================
#updated: added to git repository - phase 4 begin


import os
from dotenv import load_dotenv

load_dotenv()  
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. check your .env file.")
print(f"API key loaded.: {api_key[:5]}...")

#langchian import
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

 

#=====================================================================
# STEP 1 - Initialise Langchain components
#=====================================================================

# the LLM - same model
llm = ChatOpenAI(
    #openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    temperature=0.0,
    max_tokens=400)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


#-------Teporary Code-------
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API KEY not found.")
else:
    print(f" API load: {api_key[:8]}...")

from langchain_core.messages import HumanMessage
test = llm.invoke([HumanMessage(content="Reply with OK only.")])
print(f"LLM TEST: {test.content}")

#------End DIAGNOSTIC-------------------


#=====================================================================
# STEP 2 - Policy Document
#=====================================================================

raw_policies = [

    #AML Policies
    "All Cash transactions excedding $10,000 must be reported to regulators via a Currency Transaction ",
    "Suspicious activity must be reported via a SAR filing within 30 days of detection regardless of transaction amount.",
    "Structring transactions to avoid reporting thresholds is a criminal offence and must be flagged immediately.",
    "Transaction monitoring systems must be reviewed and updated at minimum annually to reflect current risk typologies.",
    "High risk customers must have their transactions reviewed by a senior AML officer before processing",

    #KYC policy
    "All new clients mist provide government issued photo identification before account opening can be completed.",
    "Corporate clients must provide beneficial ownership information for all shareholders holding more than 25 precent.",
    "KYC documentation must be refreshed every two years for standard risk clients and annually  for high risk clients.",
    "Enhanced due diligence is required for politicially exposed persons regardless of transaction volume.",
    "Clients who fail to provide KYC documents within 30 days must have their accounts suspended pending review."

    #Settlement Policues
    "Failed settlements must be reported to the operations manager within one hour of the settlement deadline.",
    "Trade settlement failures exceeding three consective days must be escalted to the head of operations",
    "All settlemnt discrepancies aboove $50,000 must be documented and reiewed by risk committee.",
    "Counterparty confiramtion must be received before any settlement instructio is marked as completed.",
    "Settlement failures caused by system errors must be logged in the incident management system within 2 hours."

    #General policies
    "All staff must complete annual AML training and pass the assessment with a minium score of 80 precent.",
    "Client compliants must be acknowledge within 24 hours and resolved within 15 bussiness days.",
    "Any potential conflict of interest must be disclosed to the compliance team before engaging with the client.",
    "Confidential client information must never be shared externally without writtern consent and legal approval.",
    "All compliance breaches must be reported to the Chief COmpliance Officer within 24 hours of discovery."
]

documents = [
    Document(
        page_content=policy,
        metadata={"source":"Policy Manual 2025"}
    )
    for policy in raw_policies
]

print(f"Loaded {len(documents)} documents")

#=====================================================================
# STEP 3 - Split documents
#=====================================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")   

#=====================================================================
# STEP 4 - Build or Load FAISS index
#=====================================================================

INDEX_PATH = "langchain_faiss_index"

if os.path.exists(INDEX_PATH):
    print(f"Loading existing index...")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True

    )
    print("Index loaded.\n")
else:
    print(f"Building new index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print("Index saved.\n")



#=====================================================================
# STEP 5 - Retriever
#=====================================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)

#=====================================================================
# STEP 5 - Componenet test
# Catches None issues 
#=====================================================================

print("Running Component checks....")

#test retriever
test_docs = retriever.invoke("suspicious transaction")
if not test_docs:
    raise ValueError("Retriever returned nothing. Delete index folder and retry.")

print(f" Retriever OK - returned {len(test_docs)} documents")

# Test LLM directly
from langchain_core.messages import HumanMessage
test_llm = llm.invoke([HumanMessage(content="Reply with the word OK only.")])
if not test_llm or not test_llm.content:
    raise ValueError("LLM return None. Check your API key and credits.")
print(f"    LLM OK - responded: {test_llm.content.strip()}")

print("All components OK.")


#=====================================================================
# STEP 6 - RAG Prompt
#=====================================================================

rag_prompt = ChatPromptTemplate.from_template("""
You are a senior compliabce assistant at bank operation team.
Answer the question using ONLY the context provided below

STRICT RULES:
- Only use information explicitly staed in the context below
- If the answer is not in the context say exactly:
    'I cannot find this the available policy documents.
    Please consult your compliance team directly.'
- Never invent facts, figures, deadlines, or policy details
- Quote the relevant policy statement where possible
- Be concise and professional
- If multiple poclicies are relevant mention all of them

CONTEXT FROM POLICY DOCUMENTS:
{context}

QUESTION: 
{question}

Answer:
""")
#=====================================================================
# STEP 7 - RAG Chain using LCEL
#=====================================================================

def format_docs(docs):
    """Formats retrieved docuemnts into numbered context string."""
    formatted =[]
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Policy{i}]: {doc.page_content}")
    return   "\n\n".join(formatted)

rag_chain =(
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

#=====================================================================
# STEP 8 - Ask Function
#=====================================================================

def ask(question):
    """Ask a question using the complete RAG pipeline."""
    answer = rag_chain.invoke(question)

    #debug
    #print(f"DEBUG type: {type(answer)}")
   # print(f"DEBUG value: {answer}")

    return answer

#=====================================================================
# STEP 9 - Test questions
#=====================================================================

if __name__ == "__main__":
    test_questions = [
        "What do I do when I see a suspicious transaction?",
        "How ofenn does KYC need to be updated?",
        "How old are you?",
        "What happen if a trade settlement fails?",
        "Do I need to report large cash deposit?"
    ] 

    print("=" *60)
    print("Langchain Policy Q&A RAG SYSTEM")
    print("=" *60)

    for question in test_questions:
        print(f"\n QUESTION:\n {question}")
        print(f"\n ANSWER:\n {ask(question)}")
        print("-" * 60)

    #=====================================================================
    # STEP 10 - Interactive mode
    #=====================================================================

    print("\n======== Interative Q&A ==========")
    print("Ask any question about banking policy.")
    print("Type 'quit' to end session\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == 'quit':
            print("Session ended.")
            break

        if question.strip() == "":
            continue

        print(f"\n ANSWER: \n {ask(question)}")
        print("-" * 60)