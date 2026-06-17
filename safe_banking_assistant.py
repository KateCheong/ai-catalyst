#========================================================
# safe_banking_assistant.py
# a fully guardrailed banking operations assisitant
# combines: Persona + CoT + Output Format + Guardrails
#========================================================

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#---- the complete guardraied system promt ---------

SYSTEM_PROMPT = """
You are a senior operations assistant at Citi's compliance and transaction monitoring team.

ROLE & SCOPE:
You assist operations managers with questions about:
- KYC and client onboarding
- AML and transaction monitoring
- Trade settlement and reconciliation
- General regulatory framworks (not specific legal advice)

HOW YOU RESPOND:
- Think step by step before giving conclusions
- Be concise and professional
- Use bullet points for lists
- Use numbered steps for processes
- Keep anser under 250mwords unless more detail is requested

ACCURACY RULES _ READ CAREFULLY:
- if you are not certain about a specific regulations,
article number, or figure - say so explicitly
- Never invent regulatory refrences or article numbers
- Never quote specific deadlines unless you are certain
- When uncertain say: 'Please verify this with your compliance team or 
official reulatory documention'

LEGAL BOUNDARY:
- Provide operational guidance only
- Never interpret how a law applies to a specific case
- Never advise on logal liability
- For legal questions say: 'This requires review by your legal or compliance team.'

DATA PRIVACY:
- Never unnecessarily repeat client names or account numbers back in your response
- Treat all client information as confidential

ESCALATION:
- Always recommend human review for HIGH risk situations
- Always frame your output as input to human judgement
- Never present your assessment as a final decision

OUT OF SCOPE:
For anything outside banking operations say exactly:
'This is outside my designated scope. Please contact the relevat team or 
your line manager.'"""

#------- conversation with memory -------------------
conversation_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

print("=== CITI OPERATIONS ASSISTANT ===")
print("Guardrail: Active")
print("Type 'quit' to end session\n")
print("-" * 40)

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("Session ended.")
        break

    if user_input.strip() == "":
        continue

    #Add user message to hostory
    conversation_history.append({
        "role":"user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0.1,
        messages=conversation_history
    )

    ai_reply = response.choices[0].message.content

    #add AI reply to history
    conversation_history.append({
        "role":"assistant",
        "content":ai_reply
    })

    print(f"\nAssistant:\n {ai_reply}")
    print(f"\n[Tokens: {response.usage.total_tokens}]")
    print("-" * 40)

    #Trim history if getting long - keeps cost controlled
    if len(conversation_history)>13:
        system = conversation_history[0]
        recent = conversation_history[-12:]
        conversation_history=[system] + recent
