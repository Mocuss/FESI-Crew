# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install system dependencies (including yq for YAML parsing)
RUN apt-get update && apt-get install -y wget \
    && wget https://github.com/mikefarah/yq/releases/download/v4.10.0/yq_linux_amd64 -O /usr/local/bin/yq \
    && chmod +x /usr/local/bin/yq

# Install Python dependencies (from requirements.txt)
RUN apt-get update && apt-get install -y nano
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (optional, if you run a web service)
EXPOSE 5000

# Command to run the preprocessing script when the container starts
CMD ["bash", "run.sh"]
