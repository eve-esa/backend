# EVE APIs

Build image from docker file

create a .env file with the variables you find in the env.example file.

```
docker build -t eve-image .
```

Run container based on image

```
docker run -d -p 8000:8000 --name eve-container --env-file .env eve-image
```

.env file

```
QDRANT_URL = ""
QDRANT_API_KEY = ""
MISTRAL_API_KEY =""
OPENAI_API_KEY = ""
HUGGINGFACEHUB_API_TOKEN = ""
RUNPOD_API_KEY = ""
```

Then access the http://localhost:8000/docs to visualize the endpoits.
