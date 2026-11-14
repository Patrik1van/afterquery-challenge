FROM python:3.12-slim
WORKDIR /sim
COPY coldchain.py recipes.py server.py finalize.py ./
EXPOSE 8000
CMD ["python", "server.py"]
