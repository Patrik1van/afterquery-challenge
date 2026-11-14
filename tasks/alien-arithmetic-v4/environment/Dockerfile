# The agent's container. It holds no secrets: the operator semantics and the
# puzzles live in the `lab` sidecar, and the expected answers live only in the
# verifier image.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*

COPY evaluate reveal /usr/local/bin/
RUN chmod +x /usr/local/bin/evaluate /usr/local/bin/reveal

ENV PYTHONIOENCODING=utf-8 LANG=C.UTF-8

WORKDIR /app
# Placeholder so the answer artifact always exists (an untouched placeholder scores 0).
RUN echo '{}' > /app/answer.json

# DEV ONLY: remove before submitting
ENV NVM_DIR=/root/.nvm
SHELL ["/bin/bash", "-c"]
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash \
    && . ~/.nvm/nvm.sh && nvm install 22 && nvm alias default 22 \
    && npm i -g opencode-ai@latest
