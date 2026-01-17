"""S3/MinIO storage service for video uploads and presigned URLs."""

import logging
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from ltx_server.config import S3Settings, get_settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for uploading files to S3/MinIO and generating presigned URLs."""

    def __init__(self, settings: S3Settings | None = None):
        """Initialize storage service with S3 settings."""
        self.settings = settings or get_settings().s3
        self._client = None

    @property
    def client(self):
        """Lazy-load S3 client."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.endpoint,
                region_name=self.settings.region,
                aws_access_key_id=self.settings.access_key_id,
                aws_secret_access_key=self.settings.secret_access_key,
                config=Config(signature_version="s3v4"),
            )
        return self._client

    def ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            self.client.head_bucket(Bucket=self.settings.bucket)
            logger.info(f"Bucket '{self.settings.bucket}' exists")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchBucket"):
                logger.info(f"Creating bucket '{self.settings.bucket}'")
                try:
                    self.client.create_bucket(
                        Bucket=self.settings.bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.settings.region},
                    )
                except ClientError:
                    # Some S3-compatible services don't need LocationConstraint
                    self.client.create_bucket(Bucket=self.settings.bucket)
            else:
                raise

    def upload_file(
        self,
        file_path: Path,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        """
        Upload a file to S3.

        Args:
            file_path: Local path to the file
            object_key: S3 object key (path in bucket)
            content_type: MIME type of the file

        Returns:
            The object key
        """
        logger.info(f"Uploading {file_path} to s3://{self.settings.bucket}/{object_key}")

        self.client.upload_file(
            str(file_path),
            self.settings.bucket,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )

        return object_key

    def upload_bytes(
        self,
        data: bytes,
        object_key: str,
        content_type: str = "video/mp4",
    ) -> str:
        """
        Upload bytes directly to S3.

        Args:
            data: Bytes to upload
            object_key: S3 object key
            content_type: MIME type

        Returns:
            The object key
        """
        logger.info(f"Uploading bytes to s3://{self.settings.bucket}/{object_key}")

        self.client.put_object(
            Bucket=self.settings.bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )

        return object_key

    def get_presigned_url(
        self,
        object_key: str,
        expiry: int | None = None,
    ) -> str:
        """
        Generate a presigned URL for downloading an object.

        Args:
            object_key: S3 object key
            expiry: URL expiry time in seconds (default from settings)

        Returns:
            Presigned URL string
        """
        expiry = expiry or self.settings.presigned_url_expiry

        url = self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.settings.bucket,
                "Key": object_key,
            },
            ExpiresIn=expiry,
        )

        return url

    def delete_object(self, object_key: str) -> None:
        """Delete an object from S3."""
        logger.info(f"Deleting s3://{self.settings.bucket}/{object_key}")
        self.client.delete_object(Bucket=self.settings.bucket, Key=object_key)

    def object_exists(self, object_key: str) -> bool:
        """Check if an object exists in S3."""
        try:
            self.client.head_object(Bucket=self.settings.bucket, Key=object_key)
            return True
        except ClientError:
            return False
