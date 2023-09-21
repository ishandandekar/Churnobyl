FROM python:3.9.18-slim-bullseye

WORKDIR /app

COPY serve-requirements.txt .
COPY api_prototype.py /app/
COPY temp-serve-config.yaml /app/

ARG WANDB_API_KEY
ENV WANDB_API_KEY $WANDB_API_KEY

RUN pip install --upgrade pip
RUN pip install -r serve-requirements.txt --no-cache-dir

CMD [ "python", "api_prototype.py"]