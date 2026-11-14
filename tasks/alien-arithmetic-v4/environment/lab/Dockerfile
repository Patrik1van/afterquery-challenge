FROM python:3.12-slim

# The lab alone holds the semantics and the puzzles; the agent's container has neither.
COPY server.py formula.py puzzles.json /lab/
RUN mkdir -p /var/log/lab

EXPOSE 8000
CMD ["python", "/lab/server.py"]
