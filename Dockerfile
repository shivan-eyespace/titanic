FROM python:3.11-slim

ENV PYTHONDONTWEITEBYTECODE 1
ENV PYTHONNUNBUFFERED 1

WORKDIR /src

# hadolint ignore=DL3013
RUN pip install --no-cache-dir -U pip pip-tools && \
  pip install --no-cache-dir pip-tools

COPY requirements.in dev-requirements.in /src/

RUN pip-compile -o requirements.txt requirements.in && \
  pip-compile -o dev-requirements.txt dev-requirements.in && \
  pip-sync requirements.txt dev-requirements.txt

RUN steamlit run main.py
