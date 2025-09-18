FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Install additional NLP tools
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set up working directory
WORKDIR /workspace

# Create non-root user
RUN useradd -m -s /bin/bash coder && \
    chown -R coder:coder /workspace

USER coder

# Default command
CMD ["/bin/bash"]
