# TCE Chat Assistant 🎓

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Thiagarajar College of Engineering (TCE)** using data scraped from the official TCE website.

Built with **Streamlit**, **LangChain**, **Pinecone** (vector database), **HuggingFace embeddings**, and **Google Gemini** as the LLM.

---

## 🖼️ Demo

![TCE Chat Assistant Demo](screenshots/demo.png)

---

## ✨ Features

- Conversational chat interface built with Streamlit
- Answers grounded in real TCE website content (departments, faculty, admissions, placements, history, and more)
- Source citations included with every answer
- Custom-styled UI with background image and chat bubbles

---

## 🛠️ Tech Stack

| Component        | Technology                          |
|-------------------|--------------------------------------|
| Frontend          | Streamlit                            |
| LLM               | Google Gemini (`gemini-3.6-flash`)   |
| Embeddings        | HuggingFace `all-MiniLM-L6-v2`       |
| Vector Database   | Pinecone (serverless)                |
| Orchestration     | LangChain                            |
| Web Scraping      | LangChain `WebBaseLoader` + BeautifulSoup |

---

## 📂 Project Structure

```
tce-chat-assistant/
├── app.py               # Main Streamlit chatbot application
├── ingest_data.py        # Scrapes TCE website & populates Pinecone index
├── requirements.txt      # Python dependencies
├── style.css              # Custom styling
├── bg.png                # Background image
├── tcelogo.png            # Assistant avatar
├── user.png               # User avatar
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/tce-chat-assistant.git
cd tce-chat-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:
- **Windows:** `.\venv\Scripts\Activate.ps1`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
PINECONE_API_KEY=your_pinecone_api_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

- Get a Pinecone API key: [pinecone.io](https://www.pinecone.io/)
- Get a Google Gemini API key: [ai.google.dev](https://ai.google.dev/)

### 5. Ingest data into Pinecone (run once)

```bash
python ingest_data.py
```

This scrapes the TCE website, chunks the content, generates embeddings, and stores them in a Pinecone index.

### 6. Run the chatbot

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Example Questions

- "Who is the founder of TCE?"
- "Tell me about the history of TCE"
- "What undergraduate programmes does TCE offer?"
- "What is the placement record at TCE?"
- "Who are the faculty members in the IT department?"

---

## 📌 Notes

- The chatbot only answers based on the ingested TCE website content — it will state when it doesn't have enough information rather than guessing.
- Re-run `ingest_data.py` only if you want to refresh the underlying data (e.g., after the TCE website content changes).

---



---

## 📄 License

This project is for educational purposes.
