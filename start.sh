#!/bin/bash

# Activate virtualenv
source /home/ec2-user/be/magure_ai_resume/venv/bin/activate

# Start Flask via Gunicorn in background
gunicorn -w 4 run:app -b 0.0.0.0:5001 &

# Start Celery worker
 celery -A celery_worker.celery worker --loglevel=info --pool=solo  &

# Start Celery Beat (optional)
celery -A celery_worker.celery worker beat --loglevel=info
