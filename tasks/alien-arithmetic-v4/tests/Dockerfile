# Separate verifier image. Harbor does not upload tests/ in separate mode,
# so the image carries the scripts and the expected answers itself.
# The agent's container never contains these files.
FROM python:3.12-slim

ENV PYTHONIOENCODING=utf-8 LANG=C.UTF-8
COPY . /tests/
RUN chmod +x /tests/test.sh
