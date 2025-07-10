# tasks/celery_tasks.py

import os
import re
import json
import uuid
import logging
import numpy as np
import faiss

from app import celery, db
from app.models import UploadedCV, JsonData
from utils.llm import json_parsing_with_openai
from utils.cv_processing import extract_text_from_pdf, extract_text_from_docx, get_paths_for_group
from cloudinary.exceptions import Error as CloudinaryError
import cloudinary.uploader
from sentence_transformers import SentenceTransformer
import  math

# Initialize logger and model
logger = logging.getLogger(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")

VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)




def create_chunks_with_metadata(chunks, filename, group):
    chunk_data = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "id": str(uuid.uuid4()),
            "chunk_index": i,
            "text": chunk,
            "source_file": filename,
            "group": group
        }
        chunk_data.append(metadata)
    return chunk_data




def split_text_into_chunks(text, max_words=200):
    words = text.strip().split()
    num_chunks = math.ceil(len(words) / max_words)
    return [
        " ".join(words[i * max_words:(i + 1) * max_words])
        for i in range(num_chunks)
    ]


def store_structured_chunk(cv_id, name,relevant_skills, skills, college, total_exp, filename, group):
    # Construct the summary-style structured text
    text = f"Skills: {', '.join(skills)}. " \
            f"Relevant Skills: {', '.join(relevant_skills)}. " \
           f"College: {', '.join(college)}. " \
           f"Total Experience: {total_exp}" \
            f"Name : {name}"

    chunks = split_text_into_chunks(text, max_words=200)
    index_path, metadata_path = get_paths_for_group(group)

    # Load or initialize FAISS index
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

    # Load or initialize metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = []

    for i, chunk_text in enumerate(chunks):
        chunk = {
            "id": str(uuid.uuid4()),
            "chunk_index": i,
            "text": chunk_text.strip(),
            "source_file": filename,
            "group": group,
            "cv_id": cv_id
        }

        # Generate embedding
        embedding = model.encode([chunk["text"]])[0]
        embedding = np.array([embedding]).astype("float32")
        index.add(embedding)

        # Append to metadata
        existing_metadata.append(chunk)

    # Save updated FAISS index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=2)

    logger.info(f"✅ Stored {len(chunks)} structured chunk(s) for CV {cv_id} in group '{group}'")
@celery.task(bind=True, max_retries=3, name="tasks.upload_to_cloudinary_task")
def upload_to_cloudinary_task(self, cv_id):
    try:
        cv = UploadedCV.query.get(cv_id)
        if not cv:
            raise Exception("CV not found")

        upload_result = cloudinary.uploader.upload(
            cv.filepath,
            resource_type="auto",
            folder="resumes"
        )

        if not upload_result.get("secure_url"):
            raise Exception("Upload failed, no secure_url in response")

        cv.cloud_url = upload_result["secure_url"]
        db.session.commit()

        logger.info(f"✅ Cloudinary upload successful for CV ID {cv_id}")
        return {"cv_id": cv_id, "cloud_url": cv.cloud_url}

    except Exception as e:
        logger.error(f"❌ Cloudinary upload failed for CV ID {cv_id}: {str(e)}")
        raise self.retry(exc=e, countdown=10)


@celery.task(bind=True, max_retries=3, name="tasks.parse_resume_task")
def parse_resume_task(self, cv_id, group_name):
    try:
        cv = UploadedCV.query.get(cv_id)
        if not cv:
            raise Exception("CV not found")

        # Extract raw text
        if cv.filepath.endswith(".pdf"):
            raw_text = extract_text_from_pdf(cv.filepath)
        elif cv.filepath.endswith(".docx"):
            raw_text = extract_text_from_docx(cv.filepath)
        else:
            raise Exception("Unsupported file type")

        # Prompt for structured parsing
        prompt = (
            "Extract the following fields in valid JSON format from the resume text below:\n"
            "- name: name of the candidate \n"
            "- email: list of emails\n"
            "- phone: list of phone numbers\n"
            "- college: list of college/university names\n"
            "- skills: list of skills\n"
            "- total_experience: string (e.g., '3 years 2 months')\n"
            "- current_company: list of current companies\n"
            "- past_company: list of past companies\n"
            "- relevant_skills: list of skills based on the candidate's area of expertise. "
            "For example, if the resume mentions 'frontend', include skills like React, Angular, HTML, CSS, etc.\n\n"
            "Resume:\n\n" + raw_text
        )

        result = json_parsing_with_openai(prompt)

        # Upsert JsonData
        existing = JsonData.query.filter_by(cv_id=cv.id).first()
        if not existing:
            existing = JsonData(cv_id=cv.id)
            db.session.add(existing)

        existing.data = result
        existing.email = result.get("email")
        existing.phone = result.get("phone")
        existing.college = result.get("college")
        existing.skills = result.get("skills")
        existing.relevant_skills = result.get("relevant_skills")
        existing.total_experience = result.get("total_experience")
        existing.current_company = result.get("current_company")
        existing.past_company = result.get("past_company")
        existing.parsed = True
        existing.attempts += 1
        existing.last_error = None
        db.session.commit()

        # Embed structured chunk
        try:
            store_structured_chunk(
                cv_id=cv.id,
                name = result.get("name"),
                skills=existing.skills or [],
                college=existing.college or [],
                total_exp=existing.total_experience or "",
                filename=cv.stored_filename,
                relevant_skills = existing.relevant_skills,
                group=group_name
            )
        except Exception as embed_error:
            logger.error(f"❌ Embedding structured chunk failed for CV {cv.id}: {embed_error}", exc_info=True)

    except Exception as e:
        logger.exception(f"❌ Resume parsing failed for CV ID {cv_id}")
        existing = JsonData.query.filter_by(cv_id=cv_id).first()
        if not existing:
            existing = JsonData(cv_id=cv_id)
            db.session.add(existing)

        existing.parsed = False
        existing.last_error = str(e)
        existing.attempts += 1
        db.session.commit()

        raise self.retry(exc=e, countdown=10)
