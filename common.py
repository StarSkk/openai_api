
import logging

from fastapi import FastAPI
from yaml import load, Loader


stream = open('config.yml', encoding='utf8')
settings = load(stream, Loader=Loader)
settings = dict(settings)
stream.close()

logging_level = logging.ERROR
if settings["debug"]:
    logging_level = logging.DEBUG
default_port = settings["port"]
default_temperature = settings["llm_args"]["temperature"]
default_top_p = settings["llm_args"]["top_p"]
default_max_tokens = settings["llm_args"]["max_tokens"]

logger = logging.getLogger("openai_api")
logger.setLevel(logging_level)
formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging_level)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('openai_api_server.log')
file_handler.setLevel(logging_level)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

app = FastAPI(debug=settings["debug"])
