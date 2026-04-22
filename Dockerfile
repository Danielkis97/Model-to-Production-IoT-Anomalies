# Model-to-Production IoT Anomalies
# IU Module DLBDSMTP01 - Project: From Model to Production
#
# Single image used for both iot-anomaly-api and iot-anomaly-sender services.
# The entrypoint trains the model on first startup if models/ is not mounted.

FROM python:3.11-slim

LABEL maintainer="Model-to-Production IoT Anomalies"
LABEL description="IU DLBDSMTP01 -- IoT Anomaly Detection Service Prototype"

# Required for matplotlib headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /project

# Install Python dependencies first so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Ensure output directories exist inside the image
RUN mkdir -p app/outputs models metadata data

EXPOSE 5000

# Default command: train if needed, then start the API.
# The sender service overrides this in docker-compose.yml.
CMD ["python", "entrypoint.py"]
