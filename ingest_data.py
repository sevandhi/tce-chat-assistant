# ingest_data.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
INDEX_NAME = "tce-website-data"
PINECONE_REGION = "us-west-2"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set in environment variables.")

urls = [
    "https://tce.edu/",
    "https://tce.edu/academics/study-at-tce/undergraduate-programmes",
    "https://tce.edu/academics/study-at-tce/postgraduate-programmes",
    "https://tce.edu/academics/departments",
    "https://tce.edu/academics/departments/architecture",
    "https://tce.edu/academics/departments/civil-engineering",
    "https://tce.edu/about/about-the-college",
    "https://tce.edu/about/founder",
    "https://tce.edu/about/principal",
    "https://tce.edu/about/history",
    "https://tce.edu/academics/departments/information-technology",
    "https://tce.edu/academics/departments/information-technology/faculty",
    "https://tce.edu/staff_profile/faculty/BECSE/cdcse.html",
    "https://tce.edu/staff_profile/faculty/BECSE/spmcse.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/smrit.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/amait.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/cjit.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/pkit.html",
    "https://tce.edu/staff_profile/faculty/BECSE/sscse.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/kvuit.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/siit.html",
    "https://tce.edu/staff_profile/faculty/BTECHIT/spmit.html",
    "https://tce.edu/academics/departments/information-technology/academics",
    "https://tce.edu/academics/departments/information-technology/special-interest-group",
    "https://tce.edu/admission/ug",
    "https://tce.edu/admission/pg",
    "https://tce.edu/placement",
    "https://tce.edu/placement/categories",
    "https://tce.edu/academics/"
]

print(f"Loading documents from {len(urls)} URLs...")
loader = WebBaseLoader(urls)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

print("Initializing HuggingFaceEmbeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"Initializing Pinecone with index: {INDEX_NAME} in region: {PINECONE_REGION}...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating new Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=embeddings.client.get_sentence_embedding_dimension(),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

print("Adding chunks to Pinecone Vector Store...")
index = pc.Index(INDEX_NAME)
vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embeddings,
    index_name=INDEX_NAME
)

print("Data ingestion complete!")
