#=========================================================
# faiss_search.py
# professional vector search using FAISS
# This is the foundation of production RAG system
#=========================================================

import os
import math
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#===============================================================
# STEP 1 - Your document library
# same policy as before
#===============================================================

POLICY_DOCUMENTS = [

    #AML Policies
    "All Cash transactions excedding $10,000 must be reported to regulators via a Vurrency Transaction ",
    "Suspicious activity must be reported via a SAR filing within 30 days of detection regardless of transaction amount.",
    "Structring transactions to avoid reporting thresholds is a criminal offence and must be flagged immediately.",
    "Transaction monitoring systems must be reviewed and updated at minimum annually to reflect current risk typologies.",

    #Settlement Policues
    "Failed settlements must be reported to the operations manager within one hour of the settlement deadline.",
    "Trade settlement failures exceeding three consective days must be escalted to the head of operations",
    "All settlemnt discrepancies aboove $50,000 must be documented and reiewed by risk committee.",
    "COunterparty confiramtion must be received before any settlement instructio is marked as completed.",

    #General policies
    "All staff must complete annual AML training and pass the assessment with a minium score of 80 precent.",
    "Client compliants must be acknowledge within 24 hours and resolved within 15 bussiness days.",
    "Any potential conflict of interest must be disclosed to the compliance team before engaging with the client.",
    "Confidential client information must never be shared externally without writtern consent and legal approval."
]

#===============================================================
# STEP 2 - Embedding function
# Same as before - nothing changes here
#===============================================================

def get_embedding(text):
    """
    Converts text to a vector using OpenAI embeddings. 
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

#===============================================================
# STEP 3 - Build the FAISS index
# This replaces your manual list-base index
#===============================================================

def build_faiss_index(documents):
    """
    Embeds all docuemnts and stores them in a FAISS index.

    Returns:
        index   : The FAISS search index
        doc_sotre: the original texts (so we can return them)
    """
    print (f"Building FAISS index for {len(documents)} documents...")

    #Embed all documents    
    all_vectors = []
    for i, doc in enumerate(documents):
        vector = get_embedding(doc)
        all_vectors.append(vector)
        print(f"  Embeded document {i+1} / {len(documents)}")
    
    # Convert to numpy array - FAISS requires this format
    # numpy is a Python library for working with arrays of numbers
    vectors_array = np.array(all_vectors, dtype=np.float32)

    # Get the dimension - how many numbers per vector (1536)
    dimension = vectors_array.shape[1]

    # Create FAISS index
    # IndexFlatIP = Flat index using Inner Product (dot product)
    # This is the simplest and most accurate FAISS index type
    index = faiss.IndexFlatIP(dimension)

    # Nominalise vectors before adding
    # This makes inner product equivalent to cosine similarity
    faiss.normalize_L2(vectors_array)

    # Add all vectors to the index
    index.add(vectors_array)

    print(f"FAISS index built. Contain {index.ntotal} vectors.\n")

    return index, documents

#=======================================================================
# Step 4 - Save and load index
# Embed once - use forever
#=======================================================================

def save_faiss_index (index, doc_store, index_file="policy.index", doc_file="policy_docs.json"):
    """Saves FAISS index and docuemnt to disk."""
    faiss.write_index(index, index_file)
    with open(doc_file, "w") as f:
        json.dump(doc_store, f)
    print(f"Index saved to {index_file}.")
    print(f"Document store saved to {doc_file}.")

def load_faiss_index (index_file="policy.index", doc_file="policy_docs.json"):
    """Loads a previous saved FAISS index from disk."""
    index = faiss.read_index(index_file)
    with open(doc_file, "r") as f:
        doc_store = json.load(f)
    print(f"Index loaded. Contains {index.ntotal} vectors.")
    return index, doc_store

#=========================================================================
# STEP 5 - Search function
# Now using FAISS instead of manual comparison
#=========================================================================

def search(query, index, doc_store, top_k=3):
    """
    Searches the FAISS index for documents most relevant to the query.
    Return top_k result with scores.
    """

    # Convert query to vector
    query_vector = get_embedding(query)

    # Convert a numpy array and normalise
    query_array = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_array)

    # Search the FAISS index
    # Returns: distances (Similarity scores) and indices (positions)
    scores, indices = index.search(query_array, top_k)

    #Build result list
    results = []
    for score, idx in zip(scores[0], indices[0]):

        # Edge casse - idx of -1 means no result found
        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "document": doc_store[idx],
            "index": int(idx)
        
        })
    
    return results

#============================================================
# STEP 6 - run it
#============================================================

# Build index (or load if already saved)
import os.path

if os.path.exists("policy.index"):
    # Load existing index - no need to re-embed
    index, doc_store = load_faiss_index()
else:
    # Build fresh index and save it
    index, doc_store = build_faiss_index(POLICY_DOCUMENTS)
    save_faiss_index(index, doc_store)

print()

# Test queries

queries = [
    "What do I do when I see a suspicious transaction?",
    "How ofenn does KYC need to be updated?",
    "How old are you?",
    "What happen if a trade settlement fails?",
    "Do I need to report large cash deposit?"
]

print ("=" * 55)
print("FAISS BANKING POLICY SEARCH ENGINE")
print("=" * 55)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 55)

    results = search(query, index, doc_store, top_k=3)

    for rank, result in enumerate(results, 1):
        #visual relevance bar
        score = result['score']
        doc = result['document']

        bar = "🀫" * int(score * 40)
        print(f"\n #{rank} Relevance: {score:.3f} {bar}")
        print(f"    {doc}")

    print()