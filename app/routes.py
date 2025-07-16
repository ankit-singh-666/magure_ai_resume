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
import re

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

def generate_unique_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))



def parse_experience_to_years(exp_str: str) -> float:
    if not exp_str or not isinstance(exp_str, str):
        return 0.0

    # Regex to find years and months
    year_match = re.search(r"(\d+)\s*year", exp_str)
    month_match = re.search(r"(\d+)\s*month", exp_str)

    years = int(year_match.group(1)) if year_match else 0
    months = int(month_match.group(1)) if month_match else 0

    return round(years + (months / 12), 2)



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


def enrich_candidate_details(candidate_details):
    if not candidate_details:
        return

    file_names = [c.get("file_name") for c in candidate_details if "file_name" in c]

    # Fetch CVs
    cvs = UploadedCV.query.filter(UploadedCV.stored_filename.in_(file_names)).all()
    cv_map = {cv.stored_filename: cv for cv in cvs}

    # Fetch corresponding JsonData
    cv_ids = [cv.id for cv in cvs]
    json_data_list = JsonData.query.filter(JsonData.cv_id.in_(cv_ids)).all()
    jd_map = {jd.cv_id: jd for jd in json_data_list}

    # Enrich each candidate
    for candidate in candidate_details:
        file_name = candidate.get("file_name")
        cv = cv_map.get(file_name)
        if not cv:
            continue
        jd = jd_map.get(cv.id)

        candidate["cv_id"] = cv.id
        candidate["comment"] = cv.comment
        candidate["commented_at"] = cv.commented_at.isoformat() if cv.commented_at else None
        candidate["email"] = jd.email if jd else []
        candidate["phone"] = jd.phone if jd else []
        candidate["college"] = jd.college if jd else []
        candidate["total_experience"] = jd.total_experience if jd else None


@api.route("/search_api", methods=["POST"])
def search_api():
    data = request.get_json()
    try:
        query = data.get("query")
        group_name = data.get("group")

        results = search_resume_matches(query, group_name)
        prompt = build_prompt(query, results)
        answer = query_with_openai_sdk(prompt)

        # Enrich candidate details if needed
        candidate_details = answer.get("candidate_details")
        summary = answer.get("summary")

        if summary not in ["1", "2"] and candidate_details:
            enrich_candidate_details(candidate_details)

        return jsonify({
            "answer": answer,
            "results": results
        }), 200

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

        if ext == '.pdf':
            raw_text = extract_text_from_pdf(file)
        elif ext == '.docx':
            raw_text = extract_text_from_docx(file)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        query = raw_text.strip()
        if not query:
            return jsonify({"error": "Could not derive search query from file"}), 400

        results = search_resume_matches(query, group_name)
        prompt = build_prompt(query, results)
        answer = query_with_openai_sdk(prompt)

        # Enrich candidate details if needed
        candidate_details = answer.get("candidate_details")
        summary = answer.get("summary")

        if summary not in ["1", "2"] and candidate_details:
            enrich_candidate_details(candidate_details)

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
    results = []

    # ─── 1. Group Filtering ───
    if group_name and group_name.lower() not in ["null", "undefined", ""]:
        group = Group.query.filter_by(name=group_name).first()
        if not group:
            return jsonify([]), 200
        cvs = UploadedCV.query.filter_by(group_id=group.id).order_by(UploadedCV.upload_time.desc()).all()
    else:
        cvs = UploadedCV.query.order_by(UploadedCV.upload_time.desc()).all()

    # ─── 2. Get JsonData for all CVs ───
    json_map = {
        jd.cv_id: jd
        for jd in JsonData.query.filter(JsonData.cv_id.in_([cv.id for cv in cvs])).all()
    }

    for cv in cvs:
        jd = json_map.get(cv.id)
        if not jd:
            continue

        # Base CV data
        cv_dict = cv.as_dict()
        cv_dict.update({
            "college": jd.college,
            "skills": jd.skills,
            "total_experience": jd.total_experience,
            "current_company": jd.current_company,
            "past_company": jd.past_company,
            "location": jd.location,
            "education": jd.education,
        })

        # ─── Filter: by cv_id ───
        if "cv_id" in data:
            if int(data["cv_id"]) != cv.id:
                continue

        # ─── Filter: by experience ───
        if "experience" in data:
            try:
                exp_str = jd.total_experience or ""
                exp = parse_experience_to_years(exp_str)
                min_exp, max_exp = float(data["experience"][0]), float(data["experience"][1])
                if not (min_exp <= exp <= max_exp):
                    continue
            except Exception as e:
                logger.warning(f"Experience parsing failed for CV {cv.id}: {e}")
                continue

        # ─── Filter: by skills ───
        if "skills" in data:
            required_skills = set([s.strip().lower() for s in data["skills"]])
            candidate_skills = set([s.strip().lower() for s in jd.skills or []])
            matched_skills = required_skills & candidate_skills

            cv_dict["total_skills_candidate"] = len(candidate_skills)
            cv_dict["matched_skills_count"] = len(matched_skills)

            if not matched_skills:
                continue

        # ─── Filter: by location ───
        if "location" in data:
            candidate_location = (jd.location or "").strip().lower()
            if candidate_location != data["location"].strip().lower():
                continue

        # ─── Filter: by education ───
        if "education" in data:
            edu_required = data["education"].strip().lower()
            edu_list = [e.strip().lower() for e in jd.education or []]
            if edu_required not in edu_list:
                continue

        # ─── Filter: by availability ───
        if "availability" in data:
            if not jd.last_working_date:
                continue  # Currently working → exclude from availability filter

            try:
                lwd = jd.last_working_date
                if isinstance(lwd, str):
                    lwd = datetime.fromisoformat(lwd)

                today = datetime.utcnow()
                delta_days = (today - lwd).days
                avail_req = data["availability"].strip().lower()

                if avail_req == "immediately":
                    if lwd > today:
                        continue
                elif avail_req == "15 days":
                    if delta_days < -15:
                        continue
                elif avail_req == "30 days":
                    if delta_days < -30:
                        continue
                elif avail_req == "45 days":
                    if delta_days < -45:
                        continue
                else:
                    continue  # unknown filter → skip

            except Exception as e:
                logger.warning(f"Availability filter failed for CV {cv.id}: {e}")
                continue

        # ─── Days available logic ───
        if jd.last_working_date:
            try:
                lwd = jd.last_working_date
                if isinstance(lwd, str):
                    lwd = datetime.fromisoformat(lwd)
                days_available = (datetime.utcnow() - lwd).days
                cv_dict["days_available"] = days_available
            except Exception:
                cv_dict["days_available"] = "Invalid date format"
        else:
            cv_dict["days_available"] = "Currently Working"

        results.append(cv_dict)

    return jsonify(results), 200

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


@api.route("/filters/meta", methods=["GET"])
def get_filter_metadata():
    try:
        all_json = JsonData.query.all()

        skill_set = set()
        location_set = set()
        education_set = set()
        college_set = set()

        for jd in all_json:
            # Skills
            if jd.skills:
                skill_set.update([s.strip().lower() for s in jd.skills if isinstance(s, str)])

            # Location
            if jd.location and isinstance(jd.location, str):
                location_set.add(jd.location.strip().lower())

            # Education
            if jd.education:
                education_set.update([e.strip().lower() for e in jd.education if isinstance(e, str)])

            # Colleges
            if jd.college:
                college_set.update([c.strip().lower() for c in jd.college if isinstance(c, str)])

        return jsonify({
            "skills": sorted(skill_set),
            "locations": sorted(location_set),
            "educations": sorted(education_set),
            "colleges": sorted(college_set),
        }), 200

    except Exception as e:
        logger.error(f"Error in /filters/meta: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
