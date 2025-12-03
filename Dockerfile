# Dockerfile
FROM python:3.11-slim

# Install ib_insync + pandas + matplotlib
RUN pip install --no-cache-dir \
    ib_insync==0.9.70 \
    pandas \
    matplotlib \
    numpy

# Create app directory
WORKDIR /app

# Copy only what we need
COPY martingale_ibkr_live.py .
COPY live_trading.py .
COPY tickers.txt .

# Create data folder (persisted via volume)
RUN mkdir -p /app/data

# Run every 15 seconds forever
CMD ["python", "-u", "martingale_ibkr_live.py"]
