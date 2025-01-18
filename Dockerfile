# Use a stable Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev gfortran \
    libblas-dev liblapack-dev libffi-dev libssl-dev \
    libxml2-dev libxslt-dev zlib1g-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file for caching pip install
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/

# Expose the Flask app port
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]

