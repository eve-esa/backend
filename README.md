# EVE APIs

Build image from docker file

```
docker build -t eve-image .
```

Run container based on image

```
docker run -d -p 8000:8000 --name eve-container --env-file .env eve-image
```
