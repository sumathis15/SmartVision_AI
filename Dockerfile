# Hugging Face Spaces: Streamlit now runs as Docker (Streamlit SDK was removed).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_HOME=/tmp/.streamlit \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY --chown=user:user . /app
USER user

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
