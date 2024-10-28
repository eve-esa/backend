import pytest
from fastapi.testclient import TestClient
import os
import tempfile as tfile
from server import app


client = TestClient(app)


@pytest.fixture
def valid_txt_file():
    valid_txt_file_content = "This is a test document.\nIt has multiple lines.\n"
    response = client.post(
        "/create_collection", json={"collection_name": "test_collection"}
    )
    assert response.status_code == 200  # Ensure creation was successful
    fd, path = tfile.mkstemp(suffix=".txt", prefix="abc")
    try:
        with os.fdopen(fd, "w") as tmpo:
            tmpo.write(valid_txt_file_content)
        yield path  # tests start here

    finally:
        os.remove(path)

    response = client.delete("/delete_collection?collection_name=test_collection")
    assert response.status_code == 200


@pytest.fixture
def invalid_txt_file():
    invalid_file_content = "This is an invalid file type."
    response = client.post(
        "/create_collection", json={"collection_name": "test_collection"}
    )
    assert response.status_code == 200  # Ensure creation was successful
    fd, path = tfile.mkstemp(suffix=".docx", prefix="abc")
    try:
        with os.fdopen(fd, "w") as tmpo:
            tmpo.write(invalid_file_content)
        yield path  # tests start here

    finally:
        os.remove(path)
    response = client.delete("/delete_collection?collection_name=test_collection")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_add_document_list_valid_file(valid_txt_file):
    data = {
        "collection_name": "test_collection",
        "embeddings_model": "text-embedding-3-small",
    }

    # Use the valid_txt_file fixture
    with open(valid_txt_file, "rb") as file:
        files = {
            "files": (valid_txt_file, file, "text/plain"),
        }

        # Send the POST request to the endpoint
        response = client.post("/add_document_list", data=data, files=files)

    # Validate the response
    assert response.status_code == 200
    assert "Documents uploaded successfully" in response.json()["message"]


@pytest.mark.asyncio
async def test_add_document_list_invalid_file(invalid_txt_file):
    data = {
        "collection_name": "test_collection",
        "embeddings_model": "text-embedding-3-small",
    }

    # Use the valid_txt_file fixture
    with open(invalid_txt_file, "rb") as file:
        files = {
            "files": (invalid_txt_file, file, "text/plain"),
        }

        # Send the POST request to the endpoint
        response = client.post("/add_document_list", data=data, files=files)

    # Validate the response
    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "No valid files were processed. Only PDF and TXT files are supported."
    )
