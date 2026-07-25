"""
S3-compatible storage connector, used for DVC remote access and pulling
raw TCGA data archives.

Hard rule: credentials NEVER appear as function arguments with real
values in calling code, and never get logged. They come from environment
variables only (set as GitHub Actions secrets in CI, or locally via
`.env` which is gitignored). If you find yourself typing an access key
into a script, stop -- that key is now in your shell history and
possibly your git diff.
"""
from __future__ import annotations

import os

import boto3

from mosaic.utils.logging import get_logger

logger = get_logger(__name__)


def get_s3_client():
    """Build an S3 client from environment variables.

    Required env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
    Optional: AWS_REGION (defaults to us-east-1).
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set in environment."
        )
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def download_file(bucket: str, key: str, local_path: str) -> None:
    client = get_s3_client()
    logger.info(f"Downloading s3://{bucket}/{key} -> {local_path}")
    client.download_file(bucket, key, local_path)


def upload_file(local_path: str, bucket: str, key: str) -> None:
    client = get_s3_client()
    logger.info(f"Uploading {local_path} -> s3://{bucket}/{key}")
    client.upload_file(local_path, bucket, key)
