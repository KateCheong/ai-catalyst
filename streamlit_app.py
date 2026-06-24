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
api_key =os.getenv("OPENAI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key= None

if not api_key:
    st.error(
        "OPENAI API KEY NOT FOUND, add to your Streamlit Cloud Secrects"
    )
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

#----Import RAG system--------------------------------
# This imposts ask() from langchain)rag.py
# The if __name__ == "__main__" block means test questions 
# do NOT run automatically on import

# ========================================================
# Cache the rag system
# @st.cache_resource means this only runs ONCE
# even though Streamlit rerun the script constantly
# without this - the FAISS index rebuilds on every interaction
# =========================================================
@st.cache_resource
def load_rag_system():
    """
    Loads the RAG system and caches it.
    Prevents rebuilding the FAISS index on every rerun.
    """
    from langchain_rag import ask
    return ask

#load the cached RAG system
ask = load_rag_system()


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
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander(
                    f"Sources used ({len(sources)} policy sections)",
                    expanded=False
                ):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        st.caption(source)
                        if i < len(sources):
                            st.divider()
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
    
    #---Generate and display assistant response-------------
    with st.chat_message("assistant"):
        
        #Show spinner while RAG works
        with st.spinner("Searching policy docuemnts..."):
            result = ask(prompt)
        
        #Extract answer and source from result dict
        answer = result.get("answer", "No answer returned")
        sources = result.get("sources", [])

        #Display anser
        st.markdown(answer)

        #--display sources citations 
        if sources:
            with st.expander(
                f"Sources used ({len(sources)} policy sections)",
                expanded= False
            ):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:**")
                    st.caption(source)
                    if i < len(sources):
                        st.divider()
        else:
            st.caption("No sources retrieved.")
        
        #-- Add to conversation history------
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources" : sources
        })

    