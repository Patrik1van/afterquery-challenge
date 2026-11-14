FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy fastmcp

COPY beamsim/ /app/beamsim/
COPY server.py /app/server.py

ENV BEAMLINE_SEED=3
ENV BEAMLINE_BUDGET=60
ENV BEAMLINE_RANDOMIZATION=medium
ENV BEAMLINE_SCORE_TOKEN=2eedb0458196b105832cd7f1cf169edc

EXPOSE 8000

CMD ["python", "server.py"]
