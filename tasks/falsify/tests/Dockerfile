# Separate verifier image; it alone carries the true operators (semantics.json).
FROM python:3.12-slim

ENV PYTHONIOENCODING=utf-8 LANG=C.UTF-8
COPY . /tests/
RUN chmod +x /tests/test.sh
