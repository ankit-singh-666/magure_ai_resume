from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app, db
from app.models import Group, UploadedCV, JsonData
from app.celery_tasks import parse_resume_task, upload_to_cloudinary_task
from utils.cv_processing import process_and_store_embeddings, delete_cv_data, extract_text_from_pdf, extract_text_from_docx
from utils.retriever import retrieve_similar_chunks
from utils.llm import build_prompt, query_with_openai_sdk, normalize_llm_response
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import random
import string
import traceback
import shutil
import logging

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

def generate_unique_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@api.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the Resume Analyzer API"}), 200

# Group routes
@api.route("/groups", methods=["GET"])
def list_groups():
    return jsonify([g.as_dict() for g in Group.query.all()]), 200

@api.route("/groups", methods=["POST"])
def create_group():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Group name required"}), 400
    if Group.query.filter_by(name=name).first():
        return jsonify({"error": "Group already exists"}), 400
    db.session.add(Group(name=name))
    db.session.commit()
    return jsonify({"message": "Group created"}), 201

@api.route("/groups/<int:group_id>", methods=["DELETE"])
def delete_group(group_id):
    group = Group.query.get_or_404(group_id)
    if group.cvs:
        return jsonify({"error": "Group has associated CVs"}), 400
    db.session.delete(group)
    db.session.commit()
    return jsonify({"message": "Group deleted"}), 200

# CV upload
@api.route("/upload_cv", methods=["POST"])
def upload_cv():
    try:
        files = request.files.getlist('cv')
        group_name = request.form.get("group")
        if not files or files == [None]:
            return jsonify({"error": "No files selected"}), 400
        if not group_name:
            return jsonify({"error": "No group selected"}), 400

        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            group_obj = Group(name=group_name)
            db.session.add(group_obj)
            db.session.commit()

        uploaded_files, errors = [], []
        for file in files:
            if file and allowed_file(file.filename):
                unique_filename = f"{generate_unique_id()}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                uploaded = UploadedCV(
                    original_filename=file.filename,
                    stored_filename=unique_filename,
                    filepath=filepath,
                    group_id=group_obj.id
                )
                db.session.add(uploaded)
                db.session.commit()

                parse_resume_task.delay(uploaded.id, group_name)
                upload_to_cloudinary_task.delay(uploaded.id)

                uploaded_files.append(uploaded.as_dict())
            else:
                errors.append({"filename": file.filename, "error": "Invalid file type"})

        return jsonify({"uploaded": uploaded_files, "errors": errors}), 200

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



# 🔁 Shared logic extracted here
def search_resume_matches(query, group_name):
    if not query:
        raise ValueError("No query provided")

    results = []
    if not group_name or group_name.lower() in ["null", "undefined", ""]:
        for grp in Group.query.all():
            results.extend(retrieve_similar_chunks(query, k=5, group=grp.name))
    else:
        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            raise LookupError(f"Group '{group_name}' not found")
        results = retrieve_similar_chunks(query, k=5, group=group_obj.name)

    return results


@api.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    try:
        query =data.get("query")
        group_name = data.get("group")
        results = search_resume_matches(query, group_name)

        prompt = build_prompt(query, results)
        answer = query_with_openai_sdk(prompt)

        return jsonify(normalize_llm_response({
            "answer": answer,
            "results": results
        })), 200

    except (ValueError, LookupError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal error", "details": str(e)}), 500


@api.route("/upload_jd", methods=["POST"])
def upload_jd():
    try:
        file = request.files.get("file")
        group_name = request.form.get("group")

        if not file:
            return jsonify({"error": "No file provided"}), 400

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()

        raw_text = None
        if ext == '.pdf':
            raw_text = extract_text_from_pdf(file)
        elif ext == '.docx':
            raw_text = extract_text_from_docx(file)

        if not raw_text:
            return jsonify({"error": "Unsupported file or failed extraction"}), 400

        query =raw_text.strip()
        if not query:
            return jsonify({"error": "Could not derive search query from file"}), 400

        results = search_resume_matches(query, group_name)
        prompt = build_prompt(query, results)
        answer = query_with_openai_sdk(prompt)

        return jsonify({
            "answer": answer,
            "results": results
        }), 200

    except (ValueError, LookupError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal error", "details": str(e)}), 500

@api.route("/cvs", methods=["POST"])
def get_cvs():
    data = request.get_json() or {}
    group_name = data.get("group")
    if not group_name or group_name.lower() in ["null", "undefined", ""]:
        uploads = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()
    else:
        group = Group.query.filter_by(name=group_name).first()
        uploads = UploadedCV.query.filter_by(group_id=group.id).order_by(UploadedCV.upload_time.desc()).all() if group else []
    return jsonify([u.as_dict() for u in uploads]), 200

@api.route("/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@api.route("/download/<int:cv_id>", methods=["GET"])
def download(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    return send_from_directory(app.config['UPLOAD_FOLDER'], cv.stored_filename, as_attachment=True, download_name=cv.original_filename)

@api.route("/clear_all", methods=["DELETE"])
def clear_all():
    UploadedCV.query.delete()
    Group.query.delete()
    db.session.commit()

    vector_dir = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
        os.makedirs(vector_dir)

    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
    os.makedirs(app.config['UPLOAD_FOLDER'])

    return jsonify({"message": "All data cleared"}), 200

@api.route("/delete/<int:cv_id>", methods=["DELETE"])
def delete(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    try:
        os.remove(cv.filepath)
        delete_cv_data(cv.stored_filename, group=cv.group_rel.name)
        db.session.delete(cv)
        db.session.commit()
        return jsonify({"message": f"Deleted {cv.original_filename}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route("/cv/<int:cv_id>/comment", methods=["POST"])
def add_comment(cv_id):
    data = request.get_json()
    comment = data.get("comment")
    if not comment:
        return jsonify({"error": "Comment is required"}), 400
    cv = UploadedCV.query.get_or_404(cv_id)
    cv.comment = comment
    cv.commented_at = datetime.now(ZoneInfo("Asia/Kolkata"))
    db.session.commit()
    return jsonify({"message": "Comment saved", "cv": cv.as_dict()}), 200

@api.route("/cv/<int:cv_id>/json", methods=["GET"])
def get_json_data(cv_id):
    json_record = JsonData.query.filter_by(cv_id=cv_id).first()
    if not json_record:
        return jsonify({"status": "processing"}), 202
    return jsonify({
        "parsed": json_record.parsed,
        "attempts": json_record.attempts,
        "data": json_record.data if json_record.parsed else None,
        "error": json_record.last_error
    }), 200
