FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy fastmcp

COPY beamsim/ /app/beamsim/
COPY server.py /app/server.py

ENV BEAMLINE_SEED=4
ENV BEAMLINE_BUDGET=60
ENV BEAMLINE_RANDOMIZATION=medium
ENV BEAMLINE_SCORE_TOKEN=3e7dc9a1ff9601598562309fb17533f5

EXPOSE 8000

CMD ["python", "server.py"]
