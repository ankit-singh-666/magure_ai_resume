import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utils.cv_processing import process_and_store_embeddings, delete_cv_data
from utils.retriever import retrieve_similar_chunks
from utils.llm import build_prompt, query_with_together_sdk, normalize_llm_response
import random
import string
import traceback
import logging
import shutil
from utils.llm import query_with_together_sdk, build_prompt_with_router,AGENT_SYSTEM_PROMPTS
from flask import Flask, request, jsonify, Blueprint, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_cors import CORS
from dotenv import load_dotenv
import json

# --- Cloudinary ---
import cloudinary
import cloudinary.uploader
import cloudinary.api
import cloudinary.utils


from utils.cv_processing import (
    process_and_store_embeddings,
    delete_cv_data,
    retrieve_similar_chunks
)


# Flask setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Cloudinary Configuration ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)
app.config['CLOUDINARY_FOLDER'] = 'cv_uploads'

# --- App Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = {'pdf', 'docx'} # Note: your processing code only handles PDF.
app.config['MAX_CONTENT_LENGTH'] = 70 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'cv_uploads.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-insecure-key')
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")





# ───── Database Models ─────
class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    cvs = db.relationship("UploadedCV", backref="group_rel", lazy=True, cascade="all, delete-orphan")
    def as_dict(self): return {"id": self.id, "name": self.name, "created_at": self.created_at.isoformat()}

class UploadedCV(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255), nullable=False)
    cloudinary_public_id = db.Column(db.String(255), nullable=False, unique=True)
    cloudinary_secure_url = db.Column(db.String(512), nullable=False)
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)
    def as_dict(self): return {"id": self.id, "original_filename": self.original_filename, "url": self.cloudinary_secure_url, "public_id": self.cloudinary_public_id, "upload_time": self.upload_time.isoformat(), "group": self.group_rel.name if self.group_rel else None}

# ───── Utils ─────
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def generate_unique_id(length=5): return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# ───── Blueprint & API Routes ─────
api = Blueprint('api', __name__)

@api.route("/", methods=["GET"])
def index(): return jsonify({"message": "Resume Analyzer API Ready"}), 200

@api.route("/groups", methods=["GET"])
def list_groups(): return jsonify([g.as_dict() for g in Group.query.all()]), 200

@api.route("/groups", methods=["POST"])
def create_group():
    data = request.get_json()
    name = data.get("name")
    if not name: return jsonify({"error": "Group name required"}), 400
    if Group.query.filter_by(name=name).first(): return jsonify({"error": "Group already exists"}), 400
    new_group = Group(name=name)
    db.session.add(new_group)
    db.session.commit()
    return jsonify(new_group.as_dict()), 201


@api.route("/groups/<int:group_id>", methods=["DELETE"])
def delete_group(group_id):
    group = Group.query.get_or_404(group_id)
    if group.cvs:
        return jsonify({"error": "Cannot delete group with CVs linked to it."}), 400
    db.session.delete(group)
    db.session.commit()
    return jsonify({"message": f"Group '{group.name}' deleted."}), 200

@api.route("/upload_cv", methods=["POST"])
def upload_cv():
    try:
        files = request.files.getlist('cv')
        group_name = request.form.get("group", "general").strip()
        if not files or all(f.filename == '' for f in files): return jsonify({"error": "No files selected."}), 400

        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            group_obj = Group(name=group_name)
            db.session.add(group_obj)
            db.session.commit()

        uploaded_files, errors = [], []
        for file in files:
            if file and allowed_file(file.filename):
                safe_filename = secure_filename(file.filename)
                public_id = f"{group_obj.name}/{generate_unique_id()}_{safe_filename}"
                try:
                    upload_result = cloudinary.uploader.upload(file, public_id=public_id, folder=app.config.get('CLOUDINARY_FOLDER'), resource_type="raw")
                    uploaded = UploadedCV(original_filename=file.filename, cloudinary_public_id=upload_result.get('public_id'), cloudinary_secure_url=upload_result.get('secure_url'), group_id=group_obj.id)
                    db.session.add(uploaded)
                    db.session.commit()
                    # Call the imported processing function
                    process_and_store_embeddings(upload_result.get('secure_url'), file.filename, upload_result.get('public_id'), group_obj.name)
                    uploaded_files.append(uploaded.as_dict())
                except Exception as upload_error:
                    db.session.rollback()
                    logger.error(f"Upload/DB error for {file.filename}: {upload_error}", exc_info=True)
                    errors.append({"filename": file.filename, "error": f"Upload failed: {str(upload_error)}"})
            elif file: errors.append({"filename": file.filename, "error": "Invalid file type"})
        return jsonify({"uploaded": uploaded_files, "errors": errors}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Error in /upload_cv: %s", traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred."}), 500

@api.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    query = data.get("query")
    group_name = data.get("group")  # Can be None or missing

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not group_name or str(group_name).lower() in ["null", "undefined", ""]:
        # Search across all groups
        # Get all group names from DB
        groups = [g.name for g in Group.query.all()]
        if not groups:
            return jsonify({"error": "No groups found for search"}), 404

        # Accumulate results from all groups
        all_results = []
        for grp in groups:
            results = retrieve_similar_chunks(query, k=5, group=grp)
            all_results.extend(results)

        # --- THIS IS THE FIX ---
        # The function call now includes the required 'agent_role' argument.
        prompt = build_prompt_with_router(
            question=query,
            retrieved_chunks=results,
            agent_role=selected_agent_role
        )

        llm_response_string = query_with_together_sdk(
            prompt=prompt,
            api_key=TOGETHER_API_KEY,
            system_message=system_message
        )

        try:
            answer_data = json.loads(llm_response_string)
        except json.JSONDecodeError:
            logger.error(f"LLM returned invalid JSON: {llm_response_string}")
            return jsonify({"error": "Failed to parse LLM response.", "raw_response": llm_response_string}), 500

        return jsonify({"results": results, "answer": answer_data}), 200

    except FileNotFoundError as e:
        logger.warning(f"Search failed: {e}")
        error_response = {
            "summary": "1",
            "message": "Could not perform search. No resumes have been processed for this group yet."
        }
        return jsonify({"error": str(e), "results": [], "answer": error_response}), 404

    except Exception as e:
        logger.error("Error in /search_api: %s", traceback.format_exc())
        return jsonify({"error": "An unexpected server error occurred.", "trace": str(e)}), 500

@api.route("/cvs", methods=["POST"])
def get_cvs():
    group_name = request.args.get("group", default=None, type=str)
    if not group_name:
        uploads = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()
    else:
        group_obj = Group.query.filter_by(name=group_name).first()
        uploads = UploadedCV.query.filter_by(group_id=group_obj.id).order_by(UploadedCV.upload_time.desc()).all() if group_obj else []
    return jsonify([upload.as_dict() for upload in uploads]), 200

@api.route("/delete/<int:cv_id>", methods=["DELETE"])
def delete(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    try:
        cloudinary.uploader.destroy(cv.cloudinary_public_id, resource_type="raw")
        # Call the imported deletion function
        delete_cv_data(cv.cloudinary_public_id, group=cv.group_rel.name)
        db.session.delete(cv)
        db.session.commit()
        return jsonify({"message": f"Deleted '{cv.original_filename}' successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting CV {cv_id}: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to delete CV: {str(e)}"}), 500

@api.route("/clear_all", methods=["DELETE"])
def clear_all():
    try:
        db.session.query(UploadedCV).delete()
        db.session.query(Group).delete()
        vector_dir = os.path.join(basedir, 'vector_store')
        if os.path.exists(vector_dir): shutil.rmtree(vector_dir)
        os.makedirs(vector_dir)
        cloudinary.api.delete_resources_by_prefix(app.config.get('CLOUDINARY_FOLDER'), resource_type="raw")
        db.session.commit()
        return jsonify({"message": "✅ Cleared all data successfully."}), 200
    except Exception as e:
        db.session.rollback()
        logger.error("Error in /clear_all: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# ───── App Runner ─────
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    with app.app_context():
        print("Creating db ");
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)