#============================================================
# few_shot_classifer.py
# classifies banking complaints consistencly using examples
#============================================================

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_complaint(compliant_text):
    """
    classifies a custom complaint using a few-shot examples.
    return consistent structured output everytime
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        temperature=0.0,
        messages=[
            # ------ system prompt sets the rule --------
            {
                "role":"system",
                "content": """"You are a banking operation classifier. 
                            Classify compliants using ONLY these categories:
                            CARD_ISSUE | TRANSFER_ISSUE | ACCOUNT_ACCESS | 
                            CHARGES_DISPUTE | OTHER

                            Always respond in exactly this format:
                            CATEGORY: [category]
                            PRIORITY: [HIGH/ MEDIUM/ LOW]
                            REASON: [one sentence max]
                            """
            },
            #--------example1----------------------
            {
                "role":"user",
                "content":"My international transfer has been pending for 5 days"
            },
            {
                "role":"assistant",
                "content":"CATEGORY: TRANSFER_ISSUE\n PRIORITY: HIGH\n REASON: Extended pending transfer suggest possible processing failure."
            },
              #--------example2----------------------
               {
                "role":"user",
                "content":"I was charged twice for same transaction on 15th March."
            },
            {
                "role":"assistant",
                "content":"CATEGORY: CHARGES_DISPUTE\n PRIORITY: MEDIUM\n REASON: Duplicate charge requires transaction review and potential refund."
            },
             #--------example3----------------------
               {
                "role":"user",
                "content":"I cannot log into my mobile app since yesterday."
            },
            {
                "role":"assistant",
                "content":"CATEGORY: ACCOUNT_ACCESS\n PRIORITY: MEDIUM\n REASON: Login failure may indicate authentication issue or account lock."
            },
            #--------Real Question---------------------
               {
                "role":"user",
                "content":compliant_text
            },
            
        ]
    )

    return response.choices[0].message.content

# ---- Test it with multiple compliants ------
test_complaints =[
    "My card was decliented at the supermarket even though I have funds.",
    "I need to dispute a $450 charge I don't recognise from last week.",
    "My Salary transfer hasn't arrived and it's been 3 business days.",
    "I forgot my PIN and now my card is blocked.",
    "The ATM took my money but didn't dispense cash."
]

print("=== COMPLAINT CLASSIFIER ===\n")
for compliant in test_complaints:
    print(f"COMPLAINT: {compliant}")
    result = classify_complaint(compliant)
    print(result)
    print("-" * 45)