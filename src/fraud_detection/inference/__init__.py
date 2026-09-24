"""Serving-side inference (FR-016; DOC-04 §3).

Must not import training, evaluation, tracking or pipeline modules, so the serving image can
ship ``src/fraud_detection/__init__.py`` and this package only (DOC-03 §11, DOC-04 §8).
"""
