# Use official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (optional, e.g., for building wheels)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy your project into the container
COPY . /app

# Install dependencies if needed
# If you use pyproject.toml (PEP 517 build)
RUN pip install --upgrade pip && \
    pip install .

# Command to run your package or app (change as needed)
CMD ["python", "-m", "archnemesis"]

