from flask import Flask, request, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import uuid
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from utils.cv_processing import process_and_store_embeddings, delete_cv_data
from utils.retriever import retrieve_similar_chunks
from utils.llm import build_prompt, query_with_together_sdk
import random
import string
from flask_cors import CORS

# ───── Flask Setup ─────
app = Flask(__name__)
CORS(app)
app.secret_key = 'axxiom'

UPLOAD_FOLDER = 'uploaded_cvs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'cv_uploads.db')
db = SQLAlchemy(app)

TOGETHER_API_KEY = "a22438751c3e28169bf6e875f7556b0e0f5c78c061d0789c80061dba6700b32b"

# ───── DB Model ─────
class UploadedCV(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255))
    stored_filename = db.Column(db.String(255), unique=True)
    filepath = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

    def as_dict(self):
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "filepath": self.filepath,
            "upload_time": self.upload_time.isoformat()
        }

# ───── Utils ─────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# ───── Routes ─────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Welcome to the Resume Analyzer API"}), 200

@app.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    results = retrieve_similar_chunks(query, k=5)
    prompt = build_prompt(query, results)
    answer = query_with_together_sdk(prompt, TOGETHER_API_KEY)

    return jsonify({
        "results": results,
        "answer": answer
    }), 200

@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    files = request.files.getlist('cv')
    if not files or files == [None]:
        return jsonify({"error": "No files selected."}), 400

    uploaded_files = []
    errors = []

    for file in files:
        if file and allowed_file(file.filename):
            random_suffix = generate_unique_id()
            original_filename = secure_filename(file.filename)
            unique_filename = f"{random_suffix}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

            file.save(filepath)

            uploaded = UploadedCV(
                original_filename=original_filename,
                stored_filename=unique_filename,
                filepath=filepath
            )
            db.session.add(uploaded)
            db.session.commit()

            process_and_store_embeddings(filepath, original_filename, unique_filename)

            uploaded_files.append(uploaded.as_dict())
        else:
            errors.append({"filename": file.filename, "error": "Invalid file type"})

    return jsonify({
        "uploaded": uploaded_files,
        "errors": errors
    }), 200

@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/cvs', methods=['GET'])
def get_cvs():
    uploads = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()
    return jsonify([upload.as_dict() for upload in uploads]), 200

@app.route('/download/<int:cv_id>', methods=['GET'])
def download(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        cv.stored_filename,
        as_attachment=True,
        download_name=cv.original_filename
    )

@app.route('/delete/<int:cv_id>', methods=['DELETE'])
def delete(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    try:
        os.remove(cv.filepath)
    except Exception as e:
        return jsonify({"error": f"File deletion error: {str(e)}"}), 500

    delete_cv_data(cv.stored_filename)
    db.session.delete(cv)
    db.session.commit()

    return jsonify({"message": f"Deleted '{cv.original_filename}'."}), 200


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)
