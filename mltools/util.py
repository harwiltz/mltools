import functools

from typing import Mapping

def flatten_hparams(hparams: dict, base_key: str='') -> dict:
    """
    Flattens nested dicts for easier viewing

    Some tools (like comet.ml) require hyperparameters to be in this format for logging

    Parameters
    ----------
    hparams: dict
        A dictionary
    base_key: str (optional)
        A prefix to prepend to hyperparameter keys

    Returns
    -------
    dict:
        A dict whose elements are not dicts
    """
    def parse_object(d: dict, key: str, base_key: str) -> dict:
        if len(base_key) > 0 :
            base_key = base_key + '.'
        if not isinstance(d[key], Mapping):
            return {f"{base_key}{key}": d[key]}
        return flatten_hparams(d[key], f"{base_key}{key}")

    return functools.reduce(
            lambda acc, key: {**acc, **parse_object(hparams, key, base_key)},
            hparams.keys(),
            {}
    )
