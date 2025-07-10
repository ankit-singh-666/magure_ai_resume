from flask import Flask, request, send_from_directory, jsonify, Blueprint
import os
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utils.cv_processing import process_and_store_embeddings, delete_cv_data, extract_text_from_pdf, extract_text_from_docx
from utils.retriever import retrieve_similar_chunks, expand_query_with_keywords
from utils.llm import build_prompt, query_with_openai_sdk, normalize_llm_response
import random
import string
from flask_cors import CORS
import logging
import traceback
from dotenv import load_dotenv
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo  # native in Python 3.9+


# Flask setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploaded_cvs')
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 70 * 1024 * 1024  # 70 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'cv_uploads.db')
db = SQLAlchemy(app)

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'fallback-insecure-key')
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ───── Models ─────
class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    cvs = db.relationship("UploadedCV", backref="group_rel", lazy=True)

    def as_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat()
        }

class UploadedCV(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255))
    stored_filename = db.Column(db.String(255), unique=True)
    filepath = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)
    comment = db.Column(db.Text, nullable=True)
    commented_at = db.Column(db.DateTime, nullable=True)

    def as_dict(self):
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "filepath": self.filepath,
            "upload_time": self.upload_time.isoformat(),
            "group": self.group_rel.name if self.group_rel else None,
            "comment": self.comment,
            "commented_at": self.commented_at.isoformat() if self.commented_at else None
        }


# ───── Utils ─────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# ───── Blueprint ─────
api = Blueprint('api', __name__)

@api.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the Resume Analyzer API"}), 200

# ───── Group APIs ─────
@api.route("/groups", methods=["GET"])
def list_groups():
    groups = Group.query.all()
    return jsonify([g.as_dict() for g in groups]), 200

@api.route("/groups", methods=["POST"])
def create_group():
    data = request.get_json()
    name = data.get("name")
    if not name:
        return jsonify({"error": "Group name required"}), 400
    if Group.query.filter_by(name=name).first():
        return jsonify({"error": "Group already exists"}), 400
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

# ───── CV Upload API ─────
@api.route("/upload_cv", methods=["POST"])
def upload_cv():
    try:
        files = request.files.getlist('cv')
        group_name = request.form.get("group")

        if not files or files == [None]:
            return jsonify({"error": "No files selected."}), 400

        if not group_name:
             return jsonify({"error": "No group selected."}), 400

        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            group_obj = Group(name=group_name)
            db.session.add(group_obj)
            db.session.commit()

        uploaded_files = []
        errors = []

        for file in files:
            if file and allowed_file(file.filename):
                unique_filename = f"{generate_unique_id()}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                uploaded = UploadedCV(
                    original_filename=file.filename,
                    stored_filename=unique_filename,
                    filepath=filepath,
                    group_id=group_obj.id,
                    comment= None,
                    commented_at = None
                )
                db.session.add(uploaded)
                db.session.commit()

                process_and_store_embeddings(filepath, file.filename, unique_filename, group_obj.name)
                uploaded_files.append(uploaded.as_dict())
            else:
                errors.append({"filename": file.filename, "error": "Invalid file type"})

        return jsonify({"uploaded": uploaded_files, "errors": errors}), 200

    except Exception as e:
        logger.error("Error in /upload_cv: %s", traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500



@api.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    query = data.get("query")
    group_name = data.get("group")  # Optional

    query = expand_query_with_keywords(query)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        if not group_name or str(group_name).lower() in ["null", "undefined", ""]:
            results = retrieve_similar_chunks(query, k=5, group=None)
            if not results:
                return jsonify({"error": "No indexes or metadata found for any group."}), 404
        else:
            group_obj = Group.query.filter_by(name=group_name).first()
            if not group_obj:
                return jsonify({"error": f"Group '{group_name}' not found"}), 404

            results = retrieve_similar_chunks(query, k=5, group=group_obj.name)

        prompt = build_prompt(query, results)
        answer = query_with_openai_sdk(prompt)

        raw_response = {
            "answer": answer,
            "results": results  # optional for UI/debugging
        }

        # ✅ If answer has candidate_details and a valid summary
        summary = answer.get("summary")
        candidate_details = answer.get("candidate_details")

        if summary and summary not in ["1", "2"] and candidate_details:
            file_names = [c["file_name"] for c in candidate_details]

            cvs = UploadedCV.query.filter(UploadedCV.stored_filename.in_(file_names)).all()
            cv_map = {
                cv.stored_filename: {
                    "comment": cv.comment,
                    "commented_at": cv.commented_at.isoformat() if cv.commented_at else None
                } for cv in cvs
            }

            for candidate in candidate_details:
                file_name = candidate["file_name"]
                comment_data = cv_map.get(file_name, {})
                candidate["comment"] = comment_data.get("comment")
                candidate["commented_at"] = comment_data.get("commented_at")

        return jsonify(raw_response), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logging.error("Unexpected error during search", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@api.route("/upload_jd", methods=["POST"])
def upload_jd():
    file = request.files.get("file")
    group_name = request.form.get("group")

    if not file:
        return jsonify({"error": "No file provided"}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    file_bytes = file
    if file_ext == ".pdf":
        raw_text = extract_text_from_pdf(file_bytes)
    elif file_ext == ".docx":
        raw_text = extract_text_from_docx(file_bytes)
    else:
        return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

    if not raw_text.strip():
        return jsonify({"error": "Could not extract any text from file"}), 400

    query = expand_query_with_keywords(raw_text.strip())

    if not query:
        return jsonify({"error": "Could not derive search query from file"}), 400

    if not group_name or group_name.lower() in ["null", "undefined", ""]:
        groups = [g.name for g in Group.query.all()]
        if not groups:
            return jsonify({"error": "No groups found"}), 404

        all_results = []
        for grp in groups:
            results = retrieve_similar_chunks(query, k=5, group=grp)
            all_results.extend(results)

        all_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)[:10]
        prompt = build_prompt(query, all_results)
        results = all_results
    else:
        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            return jsonify({"error": f"Group '{group_name}' not found"}), 404

        results = retrieve_similar_chunks(query, k=5, group=group_obj.name)
        prompt = build_prompt(query, results)

    try:
        answer = query_with_openai_sdk(prompt)
    except Exception as e:
        logging.error(f"LLM error: {e}")
        return jsonify({"error": "LLM failed"}), 500

    raw_response = {
        "answer": answer,
        "results": results
    }

    # ✅ Try to inject comment/commented_at
    try:
        normalized_response = normalize_llm_response(raw_response)

        summary = normalized_response.get("summary")
        candidate_details = normalized_response.get("candidate_details")

        if summary not in ["1", "2"] and candidate_details:
            file_names = [c["file_name"] for c in candidate_details]

            cvs = UploadedCV.query.filter(UploadedCV.stored_filename.in_(file_names)).all()
            cv_map = {
                cv.stored_filename: {
                    "comment": cv.comment,
                    "commented_at": cv.commented_at.isoformat() if cv.commented_at else None
                } for cv in cvs
            }

            for candidate in candidate_details:
                file_name = candidate["file_name"]
                comment_data = cv_map.get(file_name, {})
                candidate["comment"] = comment_data.get("comment")
                candidate["commented_at"] = comment_data.get("commented_at")

        return jsonify(normalized_response), 200

    except Exception as inject_error:
        logging.error(f"Post-processing error: {inject_error}", exc_info=True)
        # Fallback to raw if LLM JSON structure was unexpected
        return jsonify(raw_response), 200

# ───── CV Listing API ─────
@api.route("/cvs", methods=["POST"])
def get_cvs():
    data = request.get_json() or {}
    group_name = data.get("group")

    if not group_name or str(group_name).lower() in ["null", "undefined", ""]:
        uploads = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()
    else:
        group_obj = Group.query.filter_by(name=group_name).first()
        if not group_obj:
            return jsonify([]), 200
        uploads = UploadedCV.query.filter_by(group_id=group_obj.id).order_by(UploadedCV.upload_time.desc()).all()

    return jsonify([upload.as_dict() for upload in uploads]), 200

# ───── Other Utility APIs ─────
@api.route("/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@api.route("/download/<int:cv_id>", methods=["GET"])
def download(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    return send_from_directory(app.config['UPLOAD_FOLDER'], cv.stored_filename, as_attachment=True, download_name=cv.original_filename)

@api.route("/clear_all", methods=["DELETE"])
def clear_all():
    try:
        UploadedCV.query.delete()
        Group.query.delete()
        db.session.commit()

        vector_dir = os.path.join(basedir, 'vector_store')
        if os.path.exists(vector_dir):
            for f in os.listdir(vector_dir):
                os.remove(os.path.join(vector_dir, f))

        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'])

        return jsonify({"message": "✅ Cleared all database entries, FAISS indexes, and uploaded files."}), 200
    except Exception as e:
        logger.error("Error in /clear_all: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/delete/<int:cv_id>", methods=["DELETE"])
def delete(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    try:
        os.remove(cv.filepath)
    except Exception as e:
        return jsonify({"error": f"File deletion error: {str(e)}"}), 500

    delete_cv_data(cv.stored_filename, group=cv.group_rel.name)
    db.session.delete(cv)
    db.session.commit()

    return jsonify({"message": f"Deleted '{cv.original_filename}'"}), 200

@api.route("/cv/<int:cv_id>/comment", methods=["POST"])
def add_or_update_comment(cv_id):
    try:
        data = request.get_json()
        comment = data.get("comment")

        if not comment:
            return jsonify({"error": "Comment is required"}), 400

        cv = UploadedCV.query.get_or_404(cv_id)
        cv.comment = comment
        cv.commented_at = datetime.now(ZoneInfo("Asia/Kolkata"))  # IST
        db.session.commit()

        logger.info(f"Comment added/updated for CV ID {cv_id}")
        return jsonify({"message": "Comment added/updated", "cv": cv.as_dict()}), 200

    except Exception as e:
        logger.error("Error in POST /cv/%s/comment: %s", cv_id, traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@api.route("/cv/<int:cv_id>/comment", methods=["DELETE"])
def delete_comment(cv_id):
    try:
        cv = UploadedCV.query.get_or_404(cv_id)

        if not cv.comment:
            return jsonify({"message": "No comment to delete"}), 200

        cv.comment = None
        cv.commented_at = None
        db.session.commit()

        logger.info(f"Comment deleted for CV ID {cv_id}")
        return jsonify({"message": "Comment deleted", "cv": cv.as_dict()}), 200

    except Exception as e:
        logger.error("Error in DELETE /cv/%s/comment: %s", cv_id, traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500



# ───── App Runner ─────
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    with app.app_context():
        print("Creating db ");
        db.create_all()
        print("Tables created")
    app.run(host='0.0.0.0', port=5001, debug=True)