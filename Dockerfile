# Use an official lightweight Python image
FROM python:3.9-slim

# Avoid Python buffering and pip warnings
ENV PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt from root
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the backend code into the container
COPY back/ /app/

# Optional: install system packages if needed (e.g., matplotlib backends)
# RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Optional: expose port if using a web interface (Streamlit, Flask, etc.)
# EXPOSE 8501

# Run the chatbot pipeline
CMD ["python", "pipeline.py"]
