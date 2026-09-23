# Chest X-ray classifier — inference API image.
# Model weights are NOT baked into this image (they're gitignored and not part
# of the repo). Mount a run directory (best_model.pt + run.json) at /app/model
# via MODEL_DIR at runtime instead — see docker-compose.yml.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY api/ ./api/

ENV MODEL_DIR=/app/model
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
