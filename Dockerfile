FROM python:3.9.19

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN \
    --mount=type=cache,target=/root/.cache \
    pip install -r requirements.txt
