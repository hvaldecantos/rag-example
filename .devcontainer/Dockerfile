FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm
# FROM mcr.microsoft.com/devcontainers/python:dev-3.12-bookworm

WORKDIR /app

# Install Docker CLI and add user to docker group
RUN apt-get update && apt-get install -y docker.io \
    && groupadd docker || true \
    && usermod -aG docker vscode

# Install uv package manager
RUN pip install uv autopep8

RUN export UV_LINK_MODE=copy
