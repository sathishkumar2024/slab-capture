FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 JOBS_DIR=/tmp/slabcap-jobs PORT=7860
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py stitch_v4.py ./
# Hugging Face Spaces run the container as user 1000 and expect port 7860
RUN useradd -m -u 1000 user && mkdir -p /tmp/slabcap-jobs && chown -R user /app /tmp/slabcap-jobs
USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
