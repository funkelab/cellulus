from pathlib import Path


def to_config(cls):
    def converter(config_dict):
        if config_dict is None:
            return

        return cls(**config_dict)

    return converter


def to_path(path):
    if path is None:
        return

    return Path(path)
