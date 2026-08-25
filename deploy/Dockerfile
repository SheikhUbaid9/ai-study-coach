# Hugging Face Spaces no longer offers a "streamlit" SDK - Streamlit apps ship as
# Docker now. This is the standard HF Spaces layout.

FROM python:3.12-slim

# Spaces run containers as uid 1000, not root. Creating that user (and installing
# as them) means the model cache and Streamlit's own scratch files land somewhere
# writable; as root they'd fail at runtime with permission errors.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user \
    # sentence-transformers downloads bge-small on first search; point its cache
    # at the writable home rather than the default root-owned location.
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# 7860 is the port Spaces expects by default.
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
