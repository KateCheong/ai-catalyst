# =================================================
# First_call.py - now with conversation memory
# Your very first OpenAI API call
# =================================================

# these two line2 import the tool we need
import os               # Lets Python read your enviornment variable
from dotenv import load_dotenv     #read .env
from openai import OpenAI           #OpenAI Python Library

#--- Step 1: Load your API key from the .env file --------
# This finds your env file and loads OPEN_API_KEY into memory

load_dotenv()

# ---- Step 2: Create the OpenAI client  -----
# this is the logging in, use key automaticially
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#--- add memory list
#---- we start with just system prompt
#---- every turn, we add the user message and the AI reply

conversation_history = [
    {
        'role': 'system',
        'content':'You are a senior AML compliabce office at a Tier 1 bank. Give precise, prefessional answers only.'
    }
]

print("Chat with GPT-4o-mini (type 'quit' to stop)\n")
print("-" * 45)

# --- Loop forever until user type 'quit' --------
while True:
    #Get input from user
    user_input = input('You: ')

    #Exit condition - handle edge case of quitting
    if user_input.lower() == "quit":
        print('Session Ended.')
        break

    #skip empty input - another edge case
    if user_input.strip() == "":
        continue

    #add message to history
    conversation_history.append({
        'role':'user',
        'content':user_input
    })


    #--- Step 3.2: send full history to the API everytime----
    # think of this as sending whatsapp msg to AI
    #send the Full histpry to API everytime
    response = client.chat.completions.create(
        model='gpt-4o-mini',  #cheap model to try
        max_tokens=200,       #Max lenght of reply, keep cost low
        temperature=0.7,       #creative level, 0= robotic, 1=creative
        messages=conversation_history
    )

    # ---- Step4: print the reply -------
    #the AI's reply is buried inside the response object, let's dig it out.
    ai_reply = response.choices[0].message.content
    conversation_history.append({
        'role':'assistant',
        'content':ai_reply
    })


    #---- Step 5: print out token usage -------
    print(f"\nGPT:{ai_reply}")
    print('\n-----Token Usage------')
    print(f'Total tokens: {response.usage.total_tokens}')
    print("-"*45)