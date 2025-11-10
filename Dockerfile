FROM python:3.12-slim-bookworm

WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn sse-starlette python-dotenv

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]