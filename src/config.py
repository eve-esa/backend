# src/config.py
import yaml
import os
from dotenv import load_dotenv
import runpod
import logging

# LOGGING
# logging.basicConfig(
#     filename="./results.log",
#     level=logging.INFO,
#     filemode="w",
#     format="%(name)s - %(levelname)s - %(message)s",
# )
# logging.info("Logging is set up and working!")


# def get_logger(name: str = __name__) -> logging.Logger:
#     return logging.getLogger(name)


load_dotenv()

# ENV VARIABLES
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

runpod.api_key = RUNPOD_API_KEY


# CONFIG YAML
class Config:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get_completion_llm_id(self):
        return self.config["runpod"]["completion_llm_id"]

    def get_timeout(self):
        return self.config["runpod"]["timeout"]
