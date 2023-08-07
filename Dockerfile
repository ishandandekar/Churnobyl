FROM python:3.9

WORKDIR /fastapi-app

COPY serve-requirements.txt .

ARG WANDB_API_KEY
ENV WANDB_API_KEY $WANDB_API_KEY

RUN pip install --upgrade pip
RUN pip install -r serve-requirements.txt --no-cache-dir

COPY ./serve ./serve

CMD ["python", "./serve/api.py"]