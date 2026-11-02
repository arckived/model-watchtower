# Stage 1: base image with dependencies installed
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: train the model during the build
FROM base AS trainer
COPY app/ app/
RUN python -m app.train

# Stage 3: final image that only serves the trained model
FROM base AS runtime
RUN useradd --create-home appuser
COPY app/ app/
COPY --from=trainer /app/model/ model/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
