# chat_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "tce-website-data"
PINECONE_REGION = "us-west-2"

# --- Validate API Keys ---
if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    st.error("Please set PINECONE_API_KEY and GOOGLE_API_KEY in your environment variables.")
    st.stop()

# --- Custom CSS for background
#  and chat avatars ---
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("bg.png");
            background-size: cover;
            background-position: center;
        }
        .stChatMessage > img {
            border-radius: 50%;
            width: 36px;
            height: 36px;
            object-fit: cover;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Vectorstore and LLM ---
@st.cache_resource
def get_vectorstore_and_llm():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0)
    return vectorstore, llm

vectorstore, llm = get_vectorstore_and_llm()
retriever = vectorstore.as_retriever()

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Thiagarajar College of Engineering (TCE) created by IT student Akash BS. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information. Always cite the source URL(s) from where you found the information, formatted as a Markdown link."),
    ("human", "Context: {context}\nQuestion: {input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Streamlit UI ---
st.set_page_config(page_title="TCE Chat Assistant", page_icon="🎓")
st.title(" TCE Chat Assistant ")
st.markdown("Ask me anything about Thiagarajar College of Engineering!")

# --- Session State for Messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show previous messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="user.png" if message["role"] == "user" else "tcelogo.png"):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt_input := st.chat_input("What do you want to know about TCE?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="user.png"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="tcelogo.png"):
        with st.spinner("Thinking..."):
            try:
                response = retrieval_chain.invoke({"input": prompt_input})
                assistant_response = response["answer"]
                source_documents = response["context"]
                unique_sources = {doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata}

                citations_text = ""
                if unique_sources:
                    citations_text = "\n\n**Sources:**\n" + "\n".join([f"- [{url}]({url})" for url in sorted(unique_sources)])

                full_response = assistant_response + citations_text
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I couldn't process that request. Please try again."
                })
