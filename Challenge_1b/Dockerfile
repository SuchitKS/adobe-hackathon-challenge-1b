# Use a slim Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt')"

# Copy the main application script
COPY main.py .

# Create directories for input, output, and collections
RUN mkdir -p /app/input /app/output
RUN mkdir -p "/app/Collection 1/PDFs"
RUN mkdir -p "/app/Collection 2/PDFs"
RUN mkdir -p "/app/Collection 3/PDFs"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# The command to run the application will be provided at runtime
ENTRYPOINT ["python", "main.py"]