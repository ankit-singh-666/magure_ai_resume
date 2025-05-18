
# 🤖 RAG-Based HR Assistant for Resume Intelligence (Flask)

An AI-powered assistant that helps HR professionals query and analyse resumes in natural language. This Flask-based web app uses Retrieval Augmented Generation (RAG) to extract relevant information from PDF resumes using vector search and LLMs.

---

## 🔍 Features

- 📄 Parses and indexes PDF resumes
- 💬 Answers natural language queries (e.g., “Who has React and Node.js experience?”)
- 🔍 Performs semantic search with vector embeddings
- 🤖 Generates context-aware answers using a Large Language Model (LLM)
- 🧑‍💼 Designed for internal talent search and skill matching

---

## 🛠️ Tech Stack

- **Python & Flask** – For backend application and routing
- **PyPDF2** – To parse and extract text from PDF resumes
- **FAISS** – For efficient vector-based semantic search
- **Together.AI** – Hosted LLM inference platform  
  - Model used: `meta-llama/Llama-3.3-70B-Instruct-Turbo-Free`

---

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/hr-rag-assistant.git
cd hr-rag-assistant
````

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

1. Add your **PDF resumes** to the `resumes/` folder.

2. Run the Flask app:

```bash
python app.py
```

3. Open your browser and go to:

```
http://localhost:5000
```

4. Ask questions like:

   * "Who has experience with Django and PostgreSQL?"
   * "Find candidates with cloud and DevOps expertise."

---

## 📁 Folder Structure

```
hr-rag-assistant/
├── app.py                  # Flask app entry point
├── resumes/                # Folder with candidate PDF resumes
├── utils/                  # Text splitting, embedding, retrieval, LLM logic
├── templates/              # HTML templates for the Flask frontend
├── static/                 # Static files (CSS, JS)
├── vector_store/           # Vector database files (e.g., FAISS index)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── technical_design.md     # System architecture and design choices
```

---

## 📘 Example Queries

* “Who has worked with machine learning and Kubernetes?”
* “Find resumes with PMP certification and agile experience.”
* “Which candidates have experience in React, Node.js, and MongoDB?”

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🙌 Contributions

Feel free to fork, enhance, and make pull requests. For major changes, please open an issue first.

---


