"""The driver keeps reading the DocumentDB connection options we deploy with.

staging and prod pass ``MONGO_PARAMS`` with TLS, a replica set, secondary reads,
retryable writes off (DocumentDB has none) and a small pool. A driver bump that
renamed, dropped or reinterpreted one of them would change how every task talks
to DocumentDB without failing a single local test against plain MongoDB, so this
builds the client from the deployed shape and reads the options back. Nothing
connects: the client is lazy until the first operation.
"""

import certifi
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReadPreference

from src.utils import helpers

pytestmark = pytest.mark.no_db

# Same keys as envs/prod/app.tfvars in eve-esa/infra; the CA path points at a
# bundle that exists in the test image instead of /code/global-bundle.pem.
DEPLOYED_PARAMS = (
    "tls=true&replicaSet=rs0&readPreference=secondaryPreferred"
    "&retryWrites=false&authSource=admin&tlsCAFile={ca}&maxPoolSize=5"
)


@pytest.fixture
def deployed_uri(monkeypatch):
    monkeypatch.setattr(helpers, "MONGO_PARAMS", DEPLOYED_PARAMS.format(ca=certifi.where()))
    monkeypatch.setattr(helpers, "MONGO_USERNAME", "eve")
    monkeypatch.setattr(helpers, "MONGO_PASSWORD", "not-a-secret")
    monkeypatch.setattr(helpers, "MONGO_HOST", "docdb.invalid")
    monkeypatch.setattr(helpers, "MONGO_PORT", 27017)
    monkeypatch.setattr(helpers, "MONGO_DATABASE", "eve")
    return helpers.get_mongodb_uri()


async def test_motor_client_reads_the_deployed_options(deployed_uri):
    client = AsyncIOMotorClient(deployed_uri)
    try:
        options = client.options
        assert options.replica_set_name == "rs0"
        assert options.read_preference == ReadPreference.SECONDARY_PREFERRED
        assert options.retry_writes is False
        assert options.pool_options.max_pool_size == 5
        assert options.pool_options._ssl_context is not None
        assert client.get_database().name == "eve"
    finally:
        client.close()
