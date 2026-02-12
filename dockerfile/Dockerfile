FROM python:3.12

RUN apt-get update && \
    apt-get install -y \
        ca-certificates \
        curl && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-mock \
    pytest-asyncio \
    pytest-timeout \
    pytest-html \
    pytest-benchmark \
    coverage \
    tox \
    flake8 \
    black \
    isort \
    mypy \
    azure-identity \
    openai

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1