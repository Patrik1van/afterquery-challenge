FROM python:3.12-slim

# Streamlit requires a non-root working directory
WORKDIR /app

# Install Python dependencies first for faster Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bioreactor simulation, agent runner, and Streamlit app
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Healthcheck for Docker Desktop status monitoring
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Bind to 0.0.0.0 so the container is reachable from your host browser
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
