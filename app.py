import os
import base64
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- set_page_config MUST be the first Streamlit command ---
st.set_page_config(page_title="TCE Chat Assistant", page_icon="🎓")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "tce-website-data"
PINECONE_REGION = "us-east-1"

if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    st.error("Please set PINECONE_API_KEY and GOOGLE_API_KEY in your environment variables.")
    st.stop()

# --- Custom CSS for background (fixed with base64) ---
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_image("bg.jpg")

st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stChatMessage > img {{
            border-radius: 50%;
            width: 36px;
            height: 36px;
            object-fit: cover;
        }}
        h1 {{
            font-weight: 800 !important;
            font-size: 3.2rem !important;
            color: #FFFFFF !important;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
            background-color: rgba(0,0,0,0.35);
            padding: 12px 20px;
            border-radius: 8px;
            display: inline-block;
        }}
        [data-testid="stMarkdownContainer"] p {{
            font-weight: 600;
            font-size: 1.4rem !important;
            color: #FFFFFF !important;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        }}
        [data-testid="stChatMessage"] {{
            background-color: rgba(0, 0, 0, 0.55) !important;
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        [data-testid="stChatMessage"] li {{
            color: #FFFFFF !important;
            font-weight: 500;
            font-size: 1.2rem !important;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
        }}
        [data-testid="stChatMessage"] strong {{
            color: #FFD700 !important;
        }}
    </style>
""", unsafe_allow_html=True)

# --- Load Vectorstore and LLM ---
@st.cache_resource
def get_vectorstore_and_llm():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0)
    return vectorstore, llm

vectorstore, llm = get_vectorstore_and_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Thiagarajar College of Engineering (TCE) created by IT student Akash BS. Answer the user's question based only on the provided context. If you cannot find the answer in the context, state that you don't have enough information. Always cite the source URL(s) from where you found the information, formatted as a Markdown link."),
    ("human", "Context: {context}\nQuestion: {input}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_text(content):
    """Normalize LLM response content to a plain string, whether it's a string or a list of content blocks."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content

st.title(" TCE Chat Assistant ")
st.markdown("Ask me anything about Thiagarajar College of Engineering!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="user.png" if message["role"] == "user" else "tcelogo.png"):
        st.markdown(message["content"])

if prompt_input := st.chat_input("What do you want to know about TCE?"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user", avatar="user.png"):
        st.markdown(prompt_input)

    with st.chat_message("assistant", avatar="tcelogo.png"):
        with st.spinner("Thinking..."):
            try:
                source_documents = retriever.invoke(prompt_input)
                context_text = format_docs(source_documents)
                formatted_prompt = prompt.format_messages(context=context_text, input=prompt_input)
                llm_response = llm.invoke(formatted_prompt)
                assistant_response = extract_text(llm_response.content)

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