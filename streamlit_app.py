#====================================================
# streamlit_app.py
# Your first Streamlit web app
# Connect to langchain_rag.pt for RAG-powered answer
#====================================================

import streamlit as st
import os 
from dotenv import load_dotenv

#-----Load env variables-----------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

#----Import RAG system--------------------------------
# This imposts ask() from langchain)rag.py
# The if __name__ == "__main__" block means test questions 
# do NOT run automatically on import

from langchain_rag import ask, INDEX_PATH, test_docs

#------Page configuration - always first line--------
st.set_page_config(
    page_title="Policy Q&A",
    page_icon="🏛️",
    layout= "wide"
)

#------Sidebar - setting panel------------------------
with st.sidebar:
    st.header("Settings")

    num_sources = st.slider(
        label="Number of sources to retrieve",
        min_value=1,
        max_value=5,
        value=3   #default
    )
    st.divider()
    #Clear conversation button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages =[]
        st.rerun()

    st.divider()

    st.caption("AI Catalyst Academy")
    st.caption("Answer sourced from policy document only.")
    st.caption("Answer verify against official documentation.")

#------Main content-----------------------------------
#------Title and description--------------------------
st.title("Policy Q&A Assistant")
st.markdown(
    "Ask question about banking policy documents."
    "Every answer includes the source policies used.")

st.info(
    "This tool porvides operational guidance only."
    "It is not legal advice. Alyways consult your"
    "compliance team for regulatory decisions.",
    icon="ℹ️"
)

st.divider()

#====================================================
# SESSION STATE - conversation memory
# Initialise message history if it does not exist yet
# This one run once - persists across reruns
#====================================================

if "messages" not in st.session_state:
    st.session_state.messages =[]

#====================================================
# DISPLAY CONVESATION HISTORY
# Loop through all previous messages and display them
# This reruns every time - rebuilding the chat from history
#====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # show resources if this was an assistant message
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources used", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.caption(f"[{i}] {source}")

#====================================================
# CHAT INPUT - The question box at bottom
# st.chat_input stay fixed at the bottom
# returns the users message when press Enter
#====================================================

if prompt := st.chat_input("Ask a policy question..."):

    #---Add user message to history-----
    st.session_state.messages.append({
        "role" : "user",
        "content" : prompt
    })

    #---Display user message------------
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #---Generate and display response----
    with st.spinner("Searching policy docuemnts..."):
        answer = ask(prompt)
    
    #---Display answer-------------------
    st.markdown(answer)

    #---Add assistant message to history--
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [] #we will add in concept 4
    })
