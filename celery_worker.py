
from app import celery
import app.celery_tasks
if __name__ == '__main__':
    with app.app_context():
        celery.start()