FROM python:3.10.10

WORKDIR /code

# Copy requirements.txt and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files from the main directory (but not directories) into /code
COPY ./*.py /code/           
# COPY ./.env /code/
COPY ./requirements.txt /code/
COPY ./config.yaml /code/

COPY ./src/ /code/src/       

# Command to run the FastAPI application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
CMD ["fastapi", "run", "server.py", "--port", "80"]
