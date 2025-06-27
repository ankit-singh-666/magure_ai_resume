# Use official Python image
FROM python:3.13.3-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files (excluding .dockerignore)
COPY . .

# Expose port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python3", "app.py"]
