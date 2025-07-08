import os
import random
import string
import shutil
import traceback
import logging
from datetime import datetime
from flask import Flask, request, send_from_directory, jsonify, Blueprint
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from utils.cv_processing import process_and_store_embeddings, delete_cv_data
from utils.retriever import retrieve_similar_chunks, expand_query_with_keywords
from utils.llm import build_prompt, query_with_openai_sdk, normalize_llm_response

# ───── Load environment variables ─────
load_dotenv()

# ───── Setup logging ─────
logging.basicConfig(
    filename="/var/log/myapp.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ───── Flask Setup ─────
app = Flask(__name__)
CORS(app)

@app.before_request
def log_request():
    logging.info(f"{request.method} {request.path} from {request.remote_addr}")

# ───── Config ─────
basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(basedir, 'uploaded_cvs')
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'MAX_CONTENT_LENGTH': 70 * 1024 * 1024,
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///' + os.path.join(basedir, 'cv_uploads.db'),
    'SECRET_KEY': os.getenv('FLASK_SECRET_KEY', 'fallback-insecure-key'),
})

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ───── Database ─────
db = SQLAlchemy(app)

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

@api.route("/groups", methods=["GET"])
def list_groups():
    try:
        groups = Group.query.all()
        return jsonify([g.as_dict() for g in groups]), 200
    except Exception as e:
        logger.error("Error in /groups GET: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/groups", methods=["POST"])
def create_group():
    try:
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
    except Exception as e:
        logger.error("Error in /groups POST: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/groups/<int:group_id>", methods=["DELETE"])
def delete_group(group_id):
    try:
        group = Group.query.get_or_404(group_id)
        if group.cvs:
            return jsonify({"error": "Cannot delete group with CVs linked to it."}), 400
        db.session.delete(group)
        db.session.commit()
        return jsonify({"message": f"Group '{group.name}' deleted."}), 200
    except Exception as e:
        logger.error("Error in /groups/<id> DELETE: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

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

        uploaded_files, errors = [], []

        for file in files:
            try:
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

                    process_and_store_embeddings(filepath, file.filename, unique_filename, group_obj.name)
                    uploaded_files.append(uploaded.as_dict())
                else:
                    errors.append({"filename": file.filename, "error": "Invalid file type"})
            except Exception as inner_e:
                logger.error("Error processing file %s: %s", file.filename, traceback.format_exc())
                errors.append({"filename": file.filename, "error": str(inner_e)})

        return jsonify({"uploaded": uploaded_files, "errors": errors}), 200
    except Exception as e:
        logger.error("Error in /upload_cv: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/cv/<int:cv_id>/comment", methods=["POST"])
def add_or_update_comment(cv_id):
    try:
        data = request.get_json()
        comment = data.get("comment")
        if not comment:
            return jsonify({"error": "Comment is required"}), 400

        cv = UploadedCV.query.get_or_404(cv_id)
        cv.comment = comment
        cv.commented_at = datetime.utcnow()
        db.session.commit()

        return jsonify({"message": "Comment added/updated", "cv": cv.as_dict()}), 200
    except Exception as e:
        logger.error("Error in /cv/<id>/comment POST: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/cv/<int:cv_id>/comment", methods=["DELETE"])
def delete_comment(cv_id):
    try:
        cv = UploadedCV.query.get_or_404(cv_id)
        cv.comment = None
        cv.commented_at = None
        db.session.commit()
        return jsonify({"message": "Comment deleted", "cv": cv.as_dict()}), 200
    except Exception as e:
        logger.error("Error in /cv/<id>/comment DELETE: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/cvs", methods=["POST"])
def get_cvs():
    try:
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
    except Exception as e:
        logger.error("Error in /cvs POST: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/delete/<int:cv_id>", methods=["DELETE"])
def delete(cv_id):
    try:
        cv = UploadedCV.query.get_or_404(cv_id)
        try:
            os.remove(cv.filepath)
        except Exception as e:
            logger.warning("File deletion failed: %s", e)

        delete_cv_data(cv.stored_filename, group=cv.group_rel.name)
        db.session.delete(cv)
        db.session.commit()

        return jsonify({"message": f"Deleted '{cv.original_filename}'"}), 200
    except Exception as e:
        logger.error("Error in /delete/<id> DELETE: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@api.route("/uploads/<filename>", methods=["GET"])
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error("Error in /uploads/<filename>: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 404

# ───── App Runner ─────
app.register_blueprint(api, url_prefix='/api')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)
