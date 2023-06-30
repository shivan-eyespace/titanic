FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONNUNBUFFERED 1

WORKDIR /src

RUN apt-get update \
  && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3013
RUN pip install --no-cache-dir -U pip pip-tools && \
  pip install --no-cache-dir pip-tools

COPY requirements.in /src/

RUN pip-compile -o requirements.txt requirements.in && \
  pip-sync requirements.txt

COPY src/ data/ /src/

ENTRYPOINT ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"]

EXPOSE 8000
