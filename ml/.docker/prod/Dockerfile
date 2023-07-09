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

RUN pip-compile -o requirements.txt requirements.in --resolver=backtracking && \
  pip-sync requirements.txt

COPY main.py data /src/

# hadolint ignore=SC2028
RUN mkdir -p /root/.streamlit && \
  echo '[general]\nemail = ""' > /root/.streamlit/credentials.toml

ENTRYPOINT ["sh", "-c", "streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0"]

EXPOSE 8501
