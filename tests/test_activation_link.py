from urllib.parse import parse_qs, urlsplit

import pytest

from src.routers.auth import build_verification_url

pytestmark = pytest.mark.no_db


def test_plus_addressed_email_survives_the_round_trip():
    url = build_verification_url("user+tag@example.com", "VYG853")
    query = parse_qs(urlsplit(url).query)
    assert query["email"] == ["user+tag@example.com"]
    assert query["code"] == ["VYG853"]


def test_plus_is_percent_encoded_in_the_raw_link():
    url = build_verification_url("user+tag@example.com", "ABC123")
    assert "user+tag" not in url
    assert "user%2Btag%40example.com" in url
