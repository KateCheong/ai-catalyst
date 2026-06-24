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
    Builds the complete RAG system from scratch.
    Cache by Streamlit - only runs once per session.
    """
    try:
        #----------Step 1 import-----------------
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.documents import Document

        #--------Step 2 Initialise modeles-------
        llm = ChatOpenAI(
            model="gpt=4o-mini",
            temperature=0.0,
            max_tokens=400
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        #-----Step 3 Policy documents-----------
        raw_policies = [

                        #AML Policies
                        "All Cash transactions excedding $10,000 must be reported to regulators via a Currency Transaction ",
                        "Suspicious activity must be reported via a SAR filing within 30 days of detection regardless of transaction amount.",
                        "Structring transactions to avoid reporting thresholds is a criminal offence and must be flagged immediately.",
                        "Transaction monitoring systems must be reviewed and updated at minimum annually to reflect current risk typologies.",
                        "High risk customers must have their transactions reviewed by a senior AML officer before processing",

                        #KYC policy
                        "All new clients must provide government issued photo identification before account opening can be completed.",
                        "Corporate clients must provide beneficial ownership information for all shareholders holding more than 25 precent.",
                        "KYC documentation must be refreshed every two years for standard risk clients and annually  for high risk clients.",
                        "Enhanced due diligence is required for politicially exposed persons regardless of transaction volume.",
                        "Clients who fail to provide KYC documents within 30 days must have their accounts suspended pending review."

                        #Settlement Policues
                        "Failed settlements must be reported to the operations manager within one hour of the settlement deadline.",
                        "Trade settlement failures exceeding three consective days must be escalted to the head of operations",
                        "All settlemnt discrepancies aboove $50,000 must be documented and reiewed by risk committee.",
                        "Counterparty confiramtion must be received before any settlement instructio is marked as completed.",
                        "Settlement failures caused by system errors must be logged in the incident management system within 2 hours."

                        #General policies
                        "All staff must complete annual AML training and pass the assessment with a minium score of 80 precent.",
                        "Client compliants must be acknowledge within 24 hours and resolved within 15 bussiness days.",
                        "Any potential conflict of interest must be disclosed to the compliance team before engaging with the client.",
                        "Confidential client information must never be shared externally without writtern consent and legal approval.",
                        "All compliance breaches must be reported to the Chief COmpliance Officer within 24 hours of discovery."
                        ]

        documents = [
            Document(
                page_content=policy,
                metadata={"source":"Policy Manual 2025"}
            )
            for policy in raw_policies
        ]

        #--------Step 4 Split docuemnts--------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        #-----Step 5 Build FAISS index------------------
        # always build fresh on Sreamlit Cloud
        # no saved index available in cloud enviornment
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(
            search_kwargs = {"k": 3}
        )

        #-----Step 6 RAG prompt--------------------------
        rag_prompt = ChatPromptTemplate.from_template("""
        You are a senior compliabce assistant at bank operation team.
        Answer the question using ONLY the context provided below

        STRICT RULES:
        - Only use information explicitly staed in the context below
        - If the answer is not in the context say exactly:
            'I cannot find this the available policy documents.
            Please consult your compliance team directly.'
        - Never invent facts, figures, deadlines, or policy details
        - Quote the relevant policy statement where possible
        - Be concise and professional
        - If multiple poclicies are relevant mention all of them

        CONTEXT FROM POLICY DOCUMENTS:
        {context}

        QUESTION: 
        {question}

        Answer:
        """)
        #-----Step7 format doc heleper-----------
        def fromat_docs(docs):
            return "\n\n".join(
                f"[Policy {i+1}]: {doc.page_content}"
                for i, doc in enumerate(docs)
            )

        #----Step 8 Build ask function-----------
        def ask(question):
            try:
                retrieved_docs = retriever.invoke(question)
                context        = format_docs(retrieved_docs)
                answer_chain   = rag_prompt| llm| StrOutputParser()

                result = answer_chain.invoke({
                    "context": context,
                    "question": question
                })

                #handle return types
                if result is None:
                    answer = "No answer returned. Please try again"
                elif isinstance(result, str):
                    answer = answer
                elif isinstance(result, dict):
                    answer = answer.get("answer") or str(answer)
                elif hasattr(result, "content"):
                    answer = answer.content
                else:
                    answer = str(answer)
                
                sources = [doc.page_content for doc in retrieved_docs]

                return{"answer": answer, "sources": sources}
            
            except Exception as e:
                return{
                    "answer": f"Error: {str(e)}",
                    "sources" :[]
                }
        return ask
    except Exception as e:
        st.error(f"Failed to load RAG system: {str(e)}")
        st.stop()

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

    