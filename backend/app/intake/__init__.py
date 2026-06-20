"""Intake package.

Purpose
-------
Prepares files before ingestion. It currently handles archive extraction and the
change-detection that powers incremental ingestion, so only new or changed files
are reprocessed.
"""
