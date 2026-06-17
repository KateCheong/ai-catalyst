#=====================================================
# similarity_search.py
# Semantic search through banking policy documents
#=====================================================

import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#=================================================================
#STEP 1 - Your document library
#In real life these would come from actual policy documents
# for now we use synthetic banking policy statements
#=================================================================

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

#=======================================================================
# STEP 2 - Embedding functions
#=======================================================================

def get_embedding(text):
    """
    Converts text to a vector using OpenAI embeddings. 
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vector_a, vector_b):
    """
    Measures how similar two vectors are.
    Return a score between 0 (unrelated) and 1 (identical meaning).
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vector_b))

    #Edge case - avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    return dot_product / (magnitude_a * magnitude_b)

#==============================================================
# Step 3 - Build the search index
# Embed all documents once and store them
# this is the "encode" step from out mental model
#==============================================================

def build_index(documents):
    """
    Converts all documents into vectors.
    Returns a list of (text, vector) pairs.
    this is only needs to run once - then you store the result
    """

    print (f"Building search index with {len(documents)} documents...")
    index = []
    
    for i, doc in enumerate(documents):
        vector = get_embedding(doc)
        index.append((doc, vector))
        print(f"  Embeded document {i+1} of {len(documents)}")

    print(f"Index complete.\n")
    return index

#===============================================================
# Step 4 - Search function
# Find the most relevant docuemnts for any query
# This is the "Search" and "retrieve" steps
#===============================================================

def search(query, index, top_k=3):
    """
    Searches the index for docuemnts most relevant to the query.
    
    Args:
        query (str): The search query.
        index (list): A list of (text, vector) pairs.
        top_k (int): The number of results to return.

    Returns:
        list: A list of the top_k most relevant documents.
    """

    #convert query into a vector
    query_vector = get_embedding(query)

    #compare query against every docuemnt in the index
    results = []
    for doc_text, doc_vector in index:
        score = cosine_similarity(query_vector, doc_vector)
        results.append((score, doc_text))
    
    #sort by similarity - highest first
    results.sort(reverse=True)

    #return top_k results
    return results[:top_k]

#===============================================================
# Step 5 - Run it
#===============================================================

# Build the index once
index = build_index(POLICY_DOCUMENTS)

# Now search with natural language questions

queries = [
    "What do I do when I see a suspicious transaction?",
    "How ofenn does KYC need to be updated?",
    "How old are you?",
    "What happen if a trade settlement fails?",
    "Do I need to report large cash deposit?"
]

print ("=" * 55)
print("BANKING POLICY SEARCH ENGINE")
print("=" * 55)

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 55)

    results = search(query, index, top_k=3)

    for rank, (score, doc) in enumerate(results, 1):
        #visual relevance bar
        bar = "🀫" * int(score * 40)
        print(f"\n #{rank} Relevance: {score:.3f} {bar}")
        print(f"    {doc}")

    print()