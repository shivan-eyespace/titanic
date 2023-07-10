FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONNUNBUFFERED 1

WORKDIR /code

# hadolint ignore=DL3008
RUN apt-get update \
  && apt-get -y --no-install-recommends install libpq-dev gcc \
  && rm -rf /var/lib/apt/lists/*

# hadolint ignore=DL3013
RUN pip install --no-cache-dir -U pip pip-tools && \
  pip install --no-cache-dir pip-tools

COPY requirements.in /code/

RUN pip-compile -o requirements.txt requirements.in --resolver=backtracking && \
  pip-sync requirements.txt

COPY src/ /code/src/
COPY data/ /code/data/

ENTRYPOINT ["sh", "-c", "uvicorn src.main:app --proxy-headers --host 0.0.0.0 --port 80"]

EXPOSE 80
