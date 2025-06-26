from flask import Flask, render_template, request, redirect, url_for, flash,send_from_directory,jsonify
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
import uuid
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

from utils.cv_processing import process_and_store_embeddings, delete_cv_data
from utils.retriever import retrieve_similar_chunks
from utils.llm import build_prompt, query_with_together_sdk
from flask import Flask
from flask_cors import CORS




TOGETHER_API_KEY = "a22438751c3e28169bf6e875f7556b0e0f5c78c061d0789c80061dba6700b32b"


# ───── Flask Setup ─────
app = Flask(__name__)
CORS(app)
app.secret_key = 'axxiom'

# ───── Upload Folder ─────
UPLOAD_FOLDER = 'uploaded_cvs'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ───── Database Setup ─────
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'cv_uploads.db')
db = SQLAlchemy(app)

# ───── DB Model ─────
class UploadedCV(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255))
    stored_filename = db.Column(db.String(255), unique=True)
    filepath = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UploadedCV {self.original_filename}>"

# ───── Utils ─────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename):
    ext = filename.rsplit('.', 1)[-1]
    return f"{uuid.uuid4().hex}.{ext}"


# ───── Routes ─────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    query = data.get("query")
    print(query)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve chunks
    results = retrieve_similar_chunks(query, k=5)

    # Generate answer from Together AI
    prompt = build_prompt(query, results)
    answer = query_with_together_sdk(prompt, TOGETHER_API_KEY)

    print(answer)


    return jsonify({
        "results": results,
        "answer": answer
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



'''
@app.route('/upload_cv', methods=['POST'])
def upload_cv():
    uploaded_files = request.files.getlist('resumes')
    results = []

    if not uploaded_files:
        return jsonify({'message': 'No files uploaded.'}), 400

    for file in uploaded_files:
        file_result = {
            'filename': file.filename,
            'status': '',
            'error': None
        }

        if file and allowed_file(file.filename):
            try:
                original_filename = secure_filename(file.filename)
                unique_filename = generate_unique_filename(original_filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

                # Save the file
                file.save(filepath)

                # Save file info to DB
                uploaded = UploadedCV(
                    original_filename=original_filename,
                    stored_filename=unique_filename,
                    filepath=filepath
                )
                db.session.add(uploaded)
                db.session.commit()

                # Process embeddings or other resume-related logic
                process_and_store_embeddings(filepath, original_filename, unique_filename)

                file_result['status'] = 'success'
            except Exception as e:
                file_result['status'] = 'error'
                file_result['error'] = str(e)
        else:
            file_result['status'] = 'rejected'
            file_result['error'] = 'Invalid file format.'

        results.append(file_result)

    # Count result types
    successful = sum(1 for r in results if r['status'] == 'success')
    rejected = sum(1 for r in results if r['status'] == 'rejected')
    failed = sum(1 for r in results if r['status'] == 'error')

    return jsonify({
        'successful': successful,
        'rejected': rejected,
        'failed': failed,
        'details': results
    })
'''

@app.route('/upload_cv', methods=['GET', 'POST'])
def upload_cv():
    return render_template('upload_cv.html')
@app.route('/cvs')
def cvs():
    uploads = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()
    return render_template('cvs.html', uploads=uploads)

@app.route('/download/<int:cv_id>')
def download(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    directory = os.path.abspath(app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory, cv.stored_filename, as_attachment=True, download_name=cv.original_filename)

@app.route('/delete/<int:cv_id>', methods=['POST'])
def delete(cv_id):
    cv = UploadedCV.query.get_or_404(cv_id)
    try:
        # Delete file from filesystem
        os.remove(cv.filepath)
    except Exception as e:
        flash(f"Error deleting file: {e}", "danger")
        return redirect(url_for('cvs'))
    
    # Delete metadata and FAISS data
    delete_cv_data(cv.stored_filename)
    
    # Delete from database
    db.session.delete(cv)
    db.session.commit()
    flash(f"Deleted '{cv.stored_filename}'.", "success")
    return redirect(url_for('cvs'))


if __name__ == '__main__':
    with app.app_context():
        print("🔧 Creating database...")
        db.create_all()
        print("✅ Done.")
    app.run(debug=True)
