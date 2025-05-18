
# 🤖 RAG-Based HR Assistant for Resume Intelligence

This project is an AI-powered HR assistant that uses **Retrieval Augmented Generation (RAG)** to help HR professionals efficiently query and analyze candidate resumes in PDF format. It enables natural language search over a resume database to instantly identify skills, experience, and qualifications.



## 🔍 Key Features

- 🧠 **Natural Language Querying**: Ask questions like “Who has experience with React and Node.js?”
- 📄 **PDF Resume Parsing**: Automatically processes resumes in PDF format.
- 🧾 **Semantic Search + LLM Integration**: Retrieves relevant chunks using vector similarity and generates answers using a Large Language Model.
- 🧰 **End-to-End Application**: Complete app with UI (Streamlit/Flask) for interactive use.



## 🛠️ Tech Stack

- Python
- PyMuPDF / pdfplumber (for PDF parsing)
- FAISS / ChromaDB (for vector search)
- OpenAI / Hugging Face Transformers (for LLMs & embeddings)
- Streamlit / Flask (UI or API)
- LangChain (optional, for RAG orchestration)



## 🚀 Installation

1. **Clone the repo**:
```bash
git clone https://github.com/your-username/hr-rag-assistant.git
cd hr-rag-assistant
````

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Add PDF resumes** to the `/resumes` folder.
2. **Run the application**:

```bash
streamlit run app.py
```

3. **Ask questions** like:

   * "Find candidates with AWS certification and cloud experience"
   * "Who has worked with Django and PostgreSQL?"

---

## 📁 Folder Structure

```
hr-rag-assistant/
├── app.py
├── resumes/                # Folder containing PDF resumes
├── vector_store/           # Vector database storage
├── utils/                  # Helper functions (e.g., parsing, chunking, embedding)
├── requirements.txt
├── README.md
└── technical_design.md     # Optional technical documentation
```

---

## 📄 Example Queries

* “Who has experience in machine learning and Python?”
* “Find candidates with project management certification”
* “Who worked with both React and Node.js?”

---

## 📘 Technical Design (Summary)

The system uses:

* **Document Ingestion**: Reads PDFs and splits into chunks.
* **Embedding Generation**: Converts chunks into vectors using LLM embeddings.
* **Vector Search**: Performs semantic retrieval using FAISS/Chroma.
* **LLM Response**: Uses GPT-style models to generate contextual answers from retrieved chunks.

For full architecture, see [`technical_design.md`](./technical_design.md).

---

## 🙌 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

```

Let me know if you're using **Flask instead of Streamlit**, or want me to generate the `technical_design.md` too!
```
