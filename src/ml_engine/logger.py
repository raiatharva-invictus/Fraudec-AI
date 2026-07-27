import logging

from logging.handlers import RotatingFileHandler

from src.ml_engine.config.settings import LOG_DIR


LOG_FILE = LOG_DIR / "fraudec.log"


logger = logging.getLogger("Fraudec")


logger.setLevel(logging.INFO)


formatter = logging.Formatter(

    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

)


file_handler = RotatingFileHandler(

    LOG_FILE,

    maxBytes=5 * 1024 * 1024,

    backupCount=5

)

file_handler.setFormatter(formatter)


console_handler = logging.StreamHandler()

console_handler.setFormatter(formatter)


logger.addHandler(file_handler)

logger.addHandler(console_handler)