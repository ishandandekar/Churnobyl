FROM python:3.9.17-alpine

WORKDIR /fastapi-app

COPY serve-requirements.txt .

RUN pip install -r requirements.txt

COPY ./serve ./serve

CMD ["python", "./serve/api.py"]