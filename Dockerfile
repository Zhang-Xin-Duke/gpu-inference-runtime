FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
EXPOSE 8080
ENTRYPOINT ["gpu-runtime-service", "--host", "0.0.0.0"]
CMD ["--device", "cuda"]
