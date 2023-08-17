from typing import List

from cellulus.utils.logger import Logger


def get_logger(keys: List[str], title: str) -> Logger:
    return Logger(keys, title)
