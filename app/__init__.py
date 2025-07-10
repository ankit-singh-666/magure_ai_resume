import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from celery_app import make_celery

load_dotenv()

app = Flask(__name__)
CORS(app)

basedir = os.path.abspath(os.path.dirname(__file__ + '/../'))
UPLOAD_FOLDER = os.path.join(basedir, 'uploaded_cvs')
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=70 * 1024 * 1024,
    SQLALCHEMY_DATABASE_URI='sqlite:///' + os.path.join(basedir, 'cv_uploads.db'),
    SECRET_KEY=os.getenv('FLASK_SECRET_KEY', 'fallback-insecure-key')
)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# DB + Celery
db = SQLAlchemy(app)
celery = make_celery(app)

# Import routes and register
from app.routes import api
app.register_blueprint(api, url_prefix='/api')