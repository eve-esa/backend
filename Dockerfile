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


# Set the default port to 8000; Heroku will override this with its own PORT value
ENV PORT=8000
ENV HOST 0.0.0.0

# Command to run the FastAPI application using uvicorn with dynamic port binding
# CMD ["uvicorn", "server:app", "--host", "$HOST", "--port", "$PORT"]
CMD ["./start.sh"]