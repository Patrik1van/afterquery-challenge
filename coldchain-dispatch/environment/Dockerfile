FROM python:3.12-slim
WORKDIR /app
COPY dispatch /usr/local/bin/dispatch
RUN chmod +x /usr/local/bin/dispatch
ENV SIM_URL=http://sim:8000
# hidden-mode shifts: the agent writes what it learned here (scored); harmless elsewhere
RUN echo "{}" > /app/beliefs.json
