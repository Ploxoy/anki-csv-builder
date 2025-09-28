FROM mcr.microsoft.com/devcontainers/python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY . .
