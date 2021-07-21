from .base import Distribution
from .normal import (
    Normal,
    MixedNormal
)

def parse_distribution(dist_type, dist_params):
    dist_type = dist_type.lower()
    if dist_type == "normal":
        return Normal.instantiate(dist_params)
    if dist_type == "mixed_normal":
        return MixedNormal.instantiate(dist_params)
    raise NotImplementedError(f"Distribution type \"{dist_type}\" not recognized")
