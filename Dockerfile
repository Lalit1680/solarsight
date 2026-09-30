# SolarSight container: Streamlit app served on port 8501.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch first. If we let `ultralytics` resolve torch on its
# own, pip grabs the CUDA build (~2 GB of nvidia wheels) which we cannot use
# inside a plain container anyway. CPU torch keeps the image ~1 GB.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# App dependencies (requirements.txt does NOT pin torch for the reason above).
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code. models/ and .streamlit/secrets.toml are mounted at runtime
# (see docker-compose.yml) so the image itself stays free of secrets.
COPY app.py .
COPY src/ src/
COPY docs/ docs/
COPY supabase/ supabase/

EXPOSE 8501

# Streamlit answers HTTP on /_stcore/health; probe it so the container
# reports healthy only when the server is actually up. Python's urllib is
# used (not curl) so nothing extra must be installed in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

# secrets.toml and models/best.pt are mounted, not baked in.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
