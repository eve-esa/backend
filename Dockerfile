FROM python:3.10.10-slim

WORKDIR /code

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files from the main directory (but not directories) into /code
COPY ./*.py /code/
# COPY ./.env /code/
COPY ./requirements.txt /code/
COPY ./config.yaml /code/
COPY ./start.sh /code/
RUN chmod +x /code/start.sh
COPY ./src/ /code/src/

ENV HOST 0.0.0.0

# CMD ["uvicorn", "server:app", "--host", "$HOST", "--port", "$PORT"]
CMD ["./start.sh"]
