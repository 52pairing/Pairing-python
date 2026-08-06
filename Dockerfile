FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 복사해 레이어 캐시를 살린다.
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

COPY app ./app

EXPOSE 8000

# 워커 수는 배포 환경에서 조정한다. LLM 호출이 I/O 대기라 워커보다 async 동시성이 먼저다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
