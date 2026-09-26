"""Replay diagram generated from the current executable state declarations."""
from functools import lru_cache
from pathlib import Path
from scripts.runtime_structure import read_structure, structure_html


@lru_cache(maxsize=1)
def structure():
    return read_structure(Path(__file__).resolve().parents[1])


def state_diagram_html(machine=None):
    return structure_html(structure(),machine)
