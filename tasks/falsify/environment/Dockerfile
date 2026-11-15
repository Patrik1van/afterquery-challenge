# Agent container: only the lab clients; the operators and puzzles live in the lab sidecar.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

COPY evaluate reveal /usr/local/bin/
RUN chmod +x /usr/local/bin/evaluate /usr/local/bin/reveal

ENV PYTHONIOENCODING=utf-8 LANG=C.UTF-8

WORKDIR /app
RUN echo '{}' > /app/answer.json
