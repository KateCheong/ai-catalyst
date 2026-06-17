#==================================================================
# Rag_system.py
# Complete RAG system for banking plocy Q&A
# Combine: FAISS search + Promt Engineering + OpenAI API
# This is capstone project
#==================================================================

import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#===============================================================
# STEP 1 - Document library
# In Production, load from real PDF files
# For now, use a synthetic policy document
#===============================================================

POLICY_DOCUMENTS = [

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

#============================================================
# STEP 2 - Embedding and FAISS FUnction 
# Carried forwad from skill 3
#============================================================

def get_embedding(text):
    """
    Converts text to a vector using OpenAI embeddings. 
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_faiss_index(documents):
    """
    Embeds all docuemnts and stores them in a FAISS index."""
    print (f"Building FAISS index for {len(documents)} documents...")

    all_vectors = []
    for i, doc in enumerate(documents):
        vector = get_embedding(doc)
        all_vectors.append(vector)
        print(f"  Embeded document {i+1} / {len(documents)}")
    
    vectors_array = np.array(all_vectors, dtype=np.float32)
    dimension = vectors_array.shape[1]

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors_array)

    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors_array)
    index.add(vectors_array)

    print(f"FAISS index built. Contain {index.ntotal} vectors.\n")

    return index, documents

def save_index(index, docs, index_file="rag.index", doc_file="rag_docs.json"):
    """save index and documents to disk"""
    faiss.write_index(index, index_file)
    with open(doc_file, "w") as f:
        json.dump(docs, f)

def load_index(index_file="rag.index", doc_file="rag_docs.json"):
    """Loads index and docuemnt from disk."""
    index = faiss.read_index(index_file)
    with open(doc_file, "r") as f:
        docs = json.load(f)
    print(f"Index loaded. Contains {index.ntotal} vectors.")
    return index, docs

def retrieve(query, index, doc_sotre, top_k=3):
    """
    Retrieves the most relevant document chunks from a query, 
    this is the R in RAG - Retrieval"""
    query_vector = get_embedding(query)
    query_array = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_array)

    scores, indices = index.search(query_array, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "document": doc_sotre[idx]})
    return results

#================================================================
# STEP 3 - RAG Prompt Template
# The bridge between retrieval and generation
# ===============================================================

RAG_PROMPT_TEMPLATE = """
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
"""

#=============================================================
# STEP 4 - The RAG function
# Combines retrieval + Prompt engineering + API Call
# This is the complete RAG pipeline in one function
#=============================================================

def ask(question, index, doc_store, top_k=3):
    """
    The Complete RAG pipeline:
    1. Retrieve relavant docuemnts
    2. Build the RAG prompt
    3. Call GPT with grounded context
    4. Return answer with sources
    
    Args:
        question: the user's natural language question
        index: the FAISS index
        doc_store: original document text
        top_k: number of chunks to retrieve

    Returns:
        Dictionary with answer and source docuemnts
    """

    #-----------STEP 1: Retrieve relevant chunks------------
    retrieved = retrieve(question, index, doc_store, top_k)

    #-----------SETP 2: Build context string----------------
    # Join retrieved chunks into a number list
    context_parts =[]
    for i, result in enumerate(retrieved, 1):
        context_parts.append(f"[Policy{i}]: {result['document']}")
    context = "\n\n".join(context_parts)

    #----------SETP 3: Build the RAG prompt-----------------
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    #----------STEP 4: call the GPT with ground context-----
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.0,
        messages=[
            {"role": "user",
             "content": prompt
            }]
        )
    answer = response.choices[0].message.content

    #--------STEP 5: Return answer with sources -----------
    return{
        "question": question,
        "answer": answer,
        "sources": retrieved,
        "tokens": response.usage.total_tokens
    }

#==========================================================
# STEP 5: Run the RAG ststem
#==========================================================

#Load or build index
if os.path.exists("rag.index"):
    index, doc_store = load_index()
else:
    index, doc_store = build_faiss_index(POLICY_DOCUMENTS)
    save_index(index, doc_store)

#-------Test wuth quetions -------------------------------
test_questions = [
    "What do I do when I see a suspicious transaction?",
    "How ofenn does KYC need to be updated?",
    "How old are you?",
    "What happen if a trade settlement fails?",
    "Do I need to report large cash deposit?"
]   

print("=" *60)
print("Policy Q&A RAG SYSTEM")
print("=" *60)

for question in test_questions:
    result = ask(question, index, doc_store, top_k=3)

    print(f"\n QUESTION:\n {result['question']}")
    print(f"\n ANSWER:\n {result['answer']}")
    print(f"\n SOURCES USED:")
    for i, source in enumerate(result['sources'], 1):
        print(f"    [{i}] Score: {source['score']:.3f}")
        print(f"    {source['document'][:70]}...")
    print(f"\n TOKENS USED: {result['tokens']}")  
    print("=" * 60)

#=======================================================
# INTERACTIVE MODE - Ask your own questions
#=======================================================

print("\n======== Interative Q&A ==========")
print("Ask any question about banking policy.")
print("Type 'quit' to end session\n")

while True:
    question = input("You: ")

    if question.lower() == 'quit':
        print("Session ended.")
        break

    if question.strip() == "":
        continue

    result = ask(question, index, doc_store, top_k=3)

    print(f"\n ANSWER: \n {result['answer']}")
    print(f"\n SOURCES USED: {len(result['sources'])} policy sources.")
    print(f"\n TOKENS USED: {result['tokens']}")
    print("=" * 60) 