# tests/test_delete_document.py

import pytest
from fastapi.testclient import TestClient
import os
import tempfile as tfile
from server import app

client = TestClient(app)


@pytest.fixture
def setup_teardown():
    # Create collection with embeddings_model
    response = client.post(
        "/create_collection",
        json={
            "collection_name": "test_collection",
            "embeddings_model": "text-embedding-3-small",
        },
    )
    assert response.status_code == 200

    # Create a temporary file
    fd, path = tfile.mkstemp(suffix=".txt", prefix="test")
    try:
        # Write some content to the temporary file
        with os.fdopen(fd, "w") as tmpo:
            tmpo.write("writing temporary doc")

        # Upload the document to the vector store
        with open(path, "rb") as file:
            upload_response = client.post(
                "/add_document_list",
                data={
                    "collection_name": "test_collection",
                    "embeddings_model": "text-embedding-3-small",
                },
                files={"files": (os.path.basename(path), file, "text/plain")},
            )
            assert upload_response.status_code == 200

        yield  # Start tests

    # Teardown
    finally:
        os.remove(path)
        response = client.delete("/delete_collection?collection_name=test_collection")
        assert response.status_code == 200


@pytest.fixture
def valid_document_list():
    return {
        "collection_name": "test_collection",
        "embeddings_model": "text-embedding-3-small",
        "document_list": ["test.txt"],
    }


def test_delete_document_list_success(setup_teardown, valid_document_list):
    # Convert the document_list to a comma-separated string for the query parameter
    document_list_query = ",".join(valid_document_list["document_list"])

    # Make the DELETE request using query parameters
    response = client.delete(
        f"/delete_document_list?collection_name={valid_document_list['collection_name']}&embeddings_model={valid_document_list['embeddings_model']}&document_list={document_list_query}"
    )

    # Assert the response status code and message
    assert response.status_code == 200
    assert response.json
