#!/bin/bash

# Activate virtualenv
source /home/ec2-user/be/magure_ai_resume/venv/bin/activate

# Start Redis (optional — only if you're running via Docker)
if ! nc -z localhost 6379; then
  echo "Starting Redis..."
  docker start redis-server 2>/dev/null || docker run -d --name redis-server -p 6379:6379 redis:7.2.4
fi

# Start Flask via Gunicorn in background
echo "Starting Gunicorn..."
gunicorn -w 4 run:app -b 0.0.0.0:5001 &

# Start Celery worker
echo "Starting Celery worker..."
celery -A celery_worker.celery worker --loglevel=info --pool=solo &

# Start Celery Beat (if you're using periodic tasks)
echo "Starting Celery beat..."
celery -A celery_worker.celery beat --loglevel=info &
