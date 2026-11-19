"""An S3 GetObject or HeadObject 404 is an expected miss, not an error span.

A real botocore client answers through botocore's Stubber, instrumented the way
the server does it, and spans go through the production tracer provider into
the SDK in memory exporter.
"""

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src import observability
from src.observability.aws import EXPECTED_MISS_ATTRIBUTE, instrument_botocore


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def instrumented(exporter):
    tp = observability.build_tracer_provider(exporter, batch=False)
    instrument_botocore(tp)
    yield
    BotocoreInstrumentor().uninstrument()
    tp.shutdown()


def _client(service: str):
    return boto3.client(
        service,
        region_name="eu-west-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )


def _one_span(exporter):
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def _call_with_error(client, method, params, code, status):
    with Stubber(client) as stub:
        stub.add_client_error(
            method, service_error_code=code, http_status_code=status, expected_params=params
        )
        with pytest.raises(ClientError):
            getattr(client, method)(**params)


KEY = {"Bucket": "eve-test", "Key": "users/u/conversations/c/workspace_state"}


@pytest.mark.parametrize(
    "method,code",
    [("get_object", "NoSuchKey"), ("head_object", "404")],
)
def test_s3_404_is_an_expected_miss(instrumented, exporter, method, code):
    _call_with_error(_client("s3"), method, KEY, code, 404)

    span = _one_span(exporter)
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes[EXPECTED_MISS_ATTRIBUTE] is True
    assert span.attributes["http.status_code"] == 404
    assert not [e for e in span.events if e.name == "exception"]


@pytest.mark.parametrize("code,status", [("AccessDenied", 403), ("InternalError", 500)])
def test_s3_other_errors_stay_errors(instrumented, exporter, code, status):
    _call_with_error(_client("s3"), "get_object", KEY, code, status)

    span = _one_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert EXPECTED_MISS_ATTRIBUTE not in span.attributes
    assert [e for e in span.events if e.name == "exception"]


def test_s3_404_on_another_operation_stays_an_error(instrumented, exporter):
    _call_with_error(
        _client("s3"), "delete_bucket", {"Bucket": "eve-test"}, "NoSuchBucket", 404
    )

    span = _one_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert EXPECTED_MISS_ATTRIBUTE not in span.attributes


def test_another_service_404_stays_an_error(instrumented, exporter):
    _call_with_error(
        _client("secretsmanager"),
        "get_secret_value",
        {"SecretId": "esa-eve/test/app/missing"},
        "ResourceNotFoundException",
        404,
    )

    span = _one_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert EXPECTED_MISS_ATTRIBUTE not in span.attributes


def test_s3_success_is_untouched(instrumented, exporter):
    client = _client("s3")
    with Stubber(client) as stub:
        stub.add_response("head_object", {"ContentLength": 3}, KEY)
        client.head_object(**KEY)

    span = _one_span(exporter)
    assert span.status.status_code is StatusCode.UNSET
    assert EXPECTED_MISS_ATTRIBUTE not in span.attributes
