FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/generate_data.py && python -m src.model_training
EXPOSE 5000
CMD ["python", "-m", "src.web_app"]
