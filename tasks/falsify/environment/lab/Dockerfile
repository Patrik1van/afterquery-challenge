FROM python:3.12-slim

COPY server.py formula.py generator.py puzzles.json semantics.json /lab/
RUN mkdir -p /var/log/lab

EXPOSE 8000
CMD ["python", "/lab/server.py"]
