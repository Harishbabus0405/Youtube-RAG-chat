# 🎥 TubeMind

**Chat with YouTube Videos using AI**

TubeMind AI is an AI-powered YouTube assistant that allows users to generate video summaries and ask contextual questions about any YouTube video using Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

* 📺 Extract YouTube video transcripts
* 📝 Generate AI-powered video summaries
* 💬 Chat with video content
* 🔍 Semantic search using FAISS Vector Database
* 🌍 Multilingual transcript support (English + Hindi)
* ⚡ Fast responses powered by Groq LLM
* 🧠 Retrieval-Augmented Generation (RAG) architecture

---

## 🏗️ Architecture

YouTube URL
↓
Transcript Extraction
↓
Chunking
↓
Embeddings
↓
FAISS Vector Store
↓
Semantic Retrieval
↓
Groq LLM
↓
Answer Generation

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI / LLM

* Groq
* LangChain

### Vector Database

* FAISS

### Embeddings

* Hugging Face Sentence Transformers

### Data Source

* YouTube Transcript API

---

## 📂 Project Structure

```text
youtube-rag-chat/
│
├── app.py
├── requirements.txt
├── .env
│
├── utils/
│   ├── youtube_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── summarizer.py
│
└── vectorstore/
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Raja-0720/youtube-rag-chat.git
cd youtube-rag-chat
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_API_KEY
```

### Run Application

```bash
streamlit run app.py
```

---

## 🎯 Example Use Cases

* Summarize long YouTube videos instantly
* Learn from educational videos faster
* Chat with podcast content
* Extract insights from interviews and webinars
* Search video knowledge using natural language

---

## 📸 Screenshots

screenshots of:

* Home Page
* <img width="1883" height="879" alt="image" src="https://github.com/user-attachments/assets/d10faf3a-350a-4ea2-b854-0c2f7c9f3a80" />

* Video Summary & Chat Interface
* <img width="1684" height="897" alt="image" src="https://github.com/user-attachments/assets/6220234f-68d5-4252-adc6-0f7e8f74a468" />


---

## 🔮 Future Improvements

* Multiple Video Knowledge Base
* Compare Multiple Videos
* Source Chunk Citations
* Chat History Persistence
* Export Chat as PDF
* User Authentication

---

## 👨‍💻 Author

**HARISH BABU S**

AI & Data Science Enthusiast

---
⭐ If you found this project useful, consider giving it a star.
