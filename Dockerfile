FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure we have the models directory
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

# Expose the port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 