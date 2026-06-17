#============================================================
# transaction_analyser.py
# Analyses transactions and returns structured JSON risk data
#============================================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyse_transaction(transaction_details):
    """
    Analyses a transaction and return structured JSON
    Output can feed directly into a risk dashboard or database.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content":"""You are a senior AML risk analyst at Citi.
                
                Analyse the transaction provided and respond ONLY in this exact JSON format:
                {
                    "risk_level": "HIGH or MEDIUM or LOW",
                    "risk_score": a number from 1 to 10,
                    "red_flags": ["flag1", "flag2"],
                    "recommended_action": "what to do next",
                    "requires_sar": true or false,
                    "resoning": "one sentence explanation"
                }

                Be Precise. Be consercationve. When in doubt, rate higher."""
            },
            {
                "role":"user",
                "content": f"Analyse this transaction: {transaction_details}"
            }
        ]
    )
    # Get the raw text reply
    raw_reply = response.choices[0].message.content

    #Parse it into a Python dictionary
    #json.loads converts JSON text into usable Python data
    risk_data = json.loads(raw_reply)
    return risk_data

# ----- Test with real scenarios ----------

transactions =[
    "Cash deposit of $9,800 - third deposit this week just below $10,000 threshold.",
    "Regular monthly salary payment of $5,200 from new employer from oversea.",
    "Wire transfer of $250,000 to newly registered company in high-risk jurisdiction."
]

print("=== TrANSACTION RISK ANALYSER ====\n")

for transaction in transactions:
    print(f"TRANSACTION: {transaction}")
    print("-" * 45)

    result = analyse_transaction(transaction)

    #access each field directly from the JSON
    print(f"Risk Level : {result['risk_level']}")
    print(f"Risk Score : {result['risk_score']}/10")
    print(f"Requires SAR : {result['requires_sar']}")
    print(f"Action : {result['recommended_action']}")
    print(f"Red Flags : {', '.join(result['red_flags'])}")
    print(f"Resoning : {result['resoning']}")
    print("=" * 45)

