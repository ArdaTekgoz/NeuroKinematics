"""Deterministic, auditable dataset generation for F0-04."""

from .factory import generate_dataset, verify_dataset

__all__ = ["generate_dataset", "verify_dataset"]
