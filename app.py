import os
import base64
import gradio as gr
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_NAME = "tce-website-data"

if not all([PINECONE_API_KEY, GOOGLE_API_KEY]):
    raise ValueError("Please set PINECONE_API_KEY and GOOGLE_API_KEY as environment variables / Space secrets.")

class FastEmbedMiniLM(Embeddings):
    """all-MiniLM-L6-v2 via ONNX (fastembed) - same vectors as sentence-transformers, without torch."""
    def __init__(self):
        self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# --- Load Vectorstore and LLM (runs once at startup) ---
embeddings = FastEmbedMiniLM()
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=GOOGLE_API_KEY, temperature=0.0)

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

# --- Core chat logic (used by Gradio's ChatInterface) ---
def respond(message, history):
    try:
        source_documents = retriever.invoke(message)
        context_text = format_docs(source_documents)
        formatted_prompt = prompt.format_messages(context=context_text, input=message)
        llm_response = llm.invoke(formatted_prompt)
        assistant_response = extract_text(llm_response.content)

        unique_sources = {doc.metadata['source'] for doc in source_documents if 'source' in doc.metadata}

        citations_text = ""
        if unique_sources:
            citations_text = "\n\n**Sources:**\n" + "\n".join([f"- [{url}]({url})" for url in sorted(unique_sources)])

        return assistant_response + citations_text
    except Exception as e:
        return f"An error occurred: {e}"

# --- Custom CSS for background + styling ---
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    bg_image = get_base64_image("bg.jpg")
    bg_css = f"""
        .gradio-container {{
            background-image: url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    """
except FileNotFoundError:
    bg_css = ""

custom_css = bg_css + """
    #title-block {
        background-color: rgba(0,0,0,0.5);
        padding: 16px 20px;
        border-radius: 10px;
        display: inline-block;
    }
    #title-block h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
        margin: 0;
    }
    #title-block p {
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        margin: 4px 0 0 0;
    }
    .message-wrap {
        background-color: rgba(0, 0, 0, 0.55) !important;
        border-radius: 12px;
    }
    .message-wrap strong {
        color: #FFD700 !important;
    }
"""

# --- Build the Gradio app ---
with gr.Blocks(title="TCE Chat Assistant") as demo:
    gr.HTML(
        """
        <div id="title-block">
            <h1>🎓 TCE Chat Assistant</h1>
            <p>Ask me anything about Thiagarajar College of Engineering!</p>
        </div>
        """
    )
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(height=500, avatar_images=("user.png", "tcelogo.png")),
        textbox=gr.Textbox(placeholder="What do you want to know about TCE?"),
        examples=[
            "Who is the founder of TCE?",
            "Tell me about the history of TCE",
            "What undergraduate programmes does TCE offer?",
            "What is the placement record at TCE?",
        ],
    )

if __name__ == "__main__":
    demo.launch(css=custom_css, server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))