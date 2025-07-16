from app import db
from datetime import datetime

class Group(db.Model):
    __tablename__ = 'group'  # ✅ Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    cvs = db.relationship("UploadedCV", backref="group_rel", lazy=True)

    def as_dict(self):
        return {"id": self.id, "name": self.name, "created_at": self.created_at.isoformat()}


class UploadedCV(db.Model):
    __tablename__ = 'uploaded_cv'  # ✅ Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    original_filename = db.Column(db.String(255))
    stored_filename = db.Column(db.String(255), unique=True)
    filepath = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)

    cloud_url = db.Column(db.String(500), nullable=True)
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
            "commented_at": self.commented_at.isoformat() if self.commented_at else None,
            "cloud_url": self.cloud_url
        }


class JsonData(db.Model):
    __tablename__ = 'json_data'  # ✅ Explicit table name

    id = db.Column(db.Integer, primary_key=True)
    cv_id = db.Column(db.Integer, db.ForeignKey('uploaded_cv.id'), nullable=False)

    data = db.Column(db.JSON)
    parsed = db.Column(db.Boolean, default=False)
    attempts = db.Column(db.Integer, default=0)
    last_error = db.Column(db.Text, nullable=True)

    total_experience = db.Column(db.String(50), nullable=True)
    skills = db.Column(db.JSON, nullable=True)
    relevant_skills = db.Column(db.JSON, nullable=True)

    email = db.Column(db.JSON, nullable=True)
    phone = db.Column(db.JSON, nullable=True)
    college = db.Column(db.JSON, nullable=True)
    current_company = db.Column(db.JSON, nullable=True)
    past_company = db.Column(db.JSON, nullable=True)

    location = db.Column(db.JSON, nullable=True)
    last_working_date = db.Column(db.JSON, nullable=True)
    education = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    cv = db.relationship("UploadedCV", backref="json_data_rel")
