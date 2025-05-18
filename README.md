
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
git clone https://github.com/hchamikadilshan/AI-HR-Assistant
````

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Move Inside Project Folder**

```bash
cd AI-HR-Assistant
```

4. **Install dependencies**:

```bash
pip install -r requirements.txt
```

5. **Get your API key from [Together.AI](https://www.together.ai/)** and paste it into `app.py`:

```python
TOGETHER_API_KEY = "Enter_your_api_key_here"
```


---

## ▶️ Usage

1. **Run the Flask app**:

```bash
python app.py
```

2. **Open your browser and go to**:

```
http://localhost:5000
```

3. **Navigate to the "Upload CVs" page** from the top navbar to upload your PDF resumes.

   You’ll see three main navigation options in the navbar:

   * **Search CVs** – Ask questions and get answers using the AI assistant
   * **Upload CVs** – Upload new candidate resumes (PDF format)
   * **View CVs** – See the list of uploaded resumes



4. **Ask questions like**:

   * "Who has experience with Django and PostgreSQL?"
   * "Find candidates with cloud and DevOps expertise."

---

## 📁 Folder Structure

```
AI-HR-Assistant/
├── app.py                  # Flask app entry point
├── uploaded_cvs/           # Folder with candidate PDF resumes
├── utils/
│   ├── cv_processing.py    #Handles PDF extraction, text chunking, embedding creation, and deletion of CV data.
│   ├── llm.py              # Builds the prompt and sends it to the Together.AI LLM API to generate responses.
│   └── retriever.py        # Performs semantic search using FAISS to find relevant text chunks based on the query.
├── templates/              # HTML templates for the Flask frontend
├── static/                 # Static files (CSS, JS)
├── vector_store/           # Vector database files 
├── cv_uploads.db           # SQLite DB for storing CV details
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


