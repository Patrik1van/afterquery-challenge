FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy fastmcp

COPY beamsim/ /app/beamsim/
COPY server.py /app/server.py

ENV BEAMLINE_SEED=1
ENV BEAMLINE_BUDGET=60
ENV BEAMLINE_RANDOMIZATION=medium
ENV BEAMLINE_SCORE_TOKEN=6eb2f74e8cc1aafdb71f34d8e7931412

EXPOSE 8000

CMD ["python", "server.py"]
