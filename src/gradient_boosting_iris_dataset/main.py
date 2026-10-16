from logging_config import setup_logging

logger = setup_logging()

logger.info("Program started")
logger.info("Loading data")

try:
    result = 10 / 0
except Exception:
    logger.exception("Something went wrong")

logger.info("Program finished")
