FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip3 install --no-cache-dir -r requirements.txt

COPY .. /app
COPY .env /app/.env

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:5000 app:app"]
