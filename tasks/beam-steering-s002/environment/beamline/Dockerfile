FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy scipy fastmcp

COPY beamsim/ /app/beamsim/
COPY server.py /app/server.py

ENV BEAMLINE_SEED=2
ENV BEAMLINE_BUDGET=60
ENV BEAMLINE_RANDOMIZATION=medium
ENV BEAMLINE_SCORE_TOKEN=c47f45fa6476f69600035f4666fef0d1

EXPOSE 8000

CMD ["python", "server.py"]
