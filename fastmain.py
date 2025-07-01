from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, uuid, random, string, traceback, asyncio
from typing import List

from utils.retriever import retrieve_similar_chunks
from utils.cv_processing import process_and_store_embeddings, delete_cv_data
from utils.llm import build_prompt, async_query_with_together_sdk

app = FastAPI()
UPLOAD_FOLDER = "uploaded_cvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {"pdf", "docx"}


def generate_unique_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@app.post("/upload_cv")
async def upload_cv(cv: List[UploadFile] = File(...), group: str = Form("general")):
    uploaded_files = []
    errors = []

    for file in cv:
        if file.filename and allowed_file(file.filename):
            original_filename = file.filename
            unique_id = generate_unique_id()
            stored_filename = f"{unique_id}_{original_filename}"
            file_path = os.path.join(UPLOAD_FOLDER, stored_filename)

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Process embedding
            await asyncio.to_thread(process_and_store_embeddings, file_path, original_filename, stored_filename, group)

            uploaded_files.append({
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "group": group
            })
        else:
            errors.append({"filename": file.filename, "error": "Invalid file type"})

    return JSONResponse(content={"uploaded": uploaded_files, "errors": errors})


@app.post("/search_api")
async def search_api(data: dict):
    query = data.get("query")
    group = data.get("group", "general")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        # Run both steps concurrently
        retrieved_chunks_task = asyncio.to_thread(retrieve_similar_chunks, query, 5, group)
        prompt_task = asyncio.create_task(build_prompt_concurrently(query, retrieved_chunks_task))

        # Wait for prompt and LLM call
        prompt, retrieved_chunks = await prompt_task
        answer = await async_query_with_together_sdk(prompt)

        return {
            "results": retrieved_chunks,
            "answer": answer
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})


# helper: build prompt with awaited retrieval
async def build_prompt_concurrently(query, retrieval_task):
    retrieved_chunks = await retrieval_task
    prompt = build_prompt(query, retrieved_chunks)
    return prompt, retrieved_chunks
