#==================================================
# first_embedding.py
# Generate your first text embedding (vector)
#==================================================

import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(admin_api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """
    Converts a place of text into a vector (list of numbers).
    uses Text-embedding-3-small - cheap and fast.
    """

    response = client.embeddings.create(
        model="text-embedding-3-small",  #embedding model
        input=text
    )

    # The vector lives here - a list of 1,536 numbers
    vector = response.data[0].embedding

    return vector

# ------Generate your first embedding ----------------
text = "Suspicious transactions must be reported immediately"

print(f"Converting this text to a vector:")
print(f"'{text}'\n")

vector = get_embedding(text)

# Show basic facts about the vector
print(f"Vector legtht : {len(vector)} numbers")
print(f" first 5 numbers : {vector[:5]}")
print(f"Last 5 numbers. : {vector[-5:]}")
print(f" Smallest number  : {min(vector):.4f}")
print(f" Largest number  : {max(vector):.4f}")

#=======================================================
# Embed multiple banking sentences and compare them
#=======================================================

print("\n=== EMBEDING MULTIPLE SENTENCES ===\n")

sentences =[
    "Suspicious transaction must be reported immediately",
    "Unusal payments should be flagged without delay",
    "KYC decouments must be renew every two years.",
    "Client identity verification is required at onboarding",
    "Quarterly revenue exceeded analyst expectaions"
]

print("Generating vectors for all sentences...")
vectors ={}
for sentence in sentences:
    vectors[sentence] = get_embedding(sentence)
    print(f"    Done: {sentence[:50]}....")

print("\nAll vectors generated.")
print(f"Each vector has {len(list(vectors.values())[0])} dimensions\n")

import math

def cosine_similarity(vector_a, vector_b):
    """
    Measures how similar two vectors are.
    Return a score between 0 (unrelated) and 1 (identical meaning).
    """

    #calculate dot product
    dot_product = sum(a * b for a, b in zip (vector_a, vector_b))

    #calculate magnitudes
    magnitude_a = math.sqrt(sum(a ** 2 for a in vector_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vector_b))

    #Avoid division by zero - edge case handling
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    return dot_product/ (magnitude_a * magnitude_b)

# ---- COmpare the first sentece against all others --------
query_sentence = sentences[0]
query_vector = vectors[query_sentence]

print(f"QUERY: '{query_sentence}'\n")
print("Similarity to all other sentences:")
print("-" * 55)

# Compare query against every other sentence
results = []
for sentence in sentences[1:]:
    score = cosine_similarity(query_vector, vectors[sentence])
    results.append((score, sentence))

#sort by similarity - highest first
results.sort(reverse=True)

for score, sentence in results:
    bar ="|" * int(score * 30)
    print(f"{score:.3f}{bar}")
    print(f"    '{sentence}'")
    print()