"""The S3 client is sized for parallel artifact reads."""

from unittest.mock import patch

from src.services import storage
from src.services.storage import StorageService


def test_client_config_carries_the_pool_size():
    with patch.object(storage.boto3, "client") as make_client:
        StorageService()._client()

    config = make_client.call_args.kwargs["config"]
    assert config.max_pool_connections == 50
