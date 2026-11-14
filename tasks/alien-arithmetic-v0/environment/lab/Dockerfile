FROM python:3.12-slim

COPY server.py /lab/server.py
RUN mkdir -p /var/log/lab

EXPOSE 8000
CMD ["python", "/lab/server.py"]
