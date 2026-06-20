#====================================================
# streamlit_app.py
# Your first Streamlit web app
#====================================================

import streamlit as st

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

#------Main content-----------------------------------
#------Title and description--------------------------
st.title("Policy Q&A Assistant")
st.markdown("Ask question about banking policy documents.")

ask_button = st.button("Ask")


#------A simple tect input-----------------------------
user_input = st.text_input(
    label= "Type your question here",
    placeholder= "What is the KYC renewal period?"
)

#-----Only respond if user typed something------------
if ask_button and user_input:
    st.write(f"You asked: **{user_input}**")
    st.write(f"Will retrieve **{num_sources}** sources.")
    st.info("RAG system coming soon - this is just interface for now.")
elif ask_button and not user_input:
    st.warning("Please type a question first.")

# Add temporarily - understanding session state
if "count" not in st.session_state:
    st.session_state.count = 0

st.session_state.count +=1
st.write(f"This page has rerun {st.session_state.count} times.")