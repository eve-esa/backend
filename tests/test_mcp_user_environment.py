"""The registry environment.yaml, as stored on a toolkit row and shown to the user.

The two documents are the files on origin/main of eve-esa/mcp-tool-registry:
``servers/effis/environment.yaml`` names headers, ``servers/cdse/environment.yaml``
names process variables. See CONTRIBUTING.md, "Create environment.yaml".
"""

import logging

import pytest

from src.database.models.mcp_server import MCPServer, ToolConfig, ToolTransport
from src.services.mcp import user_credentials as uc
from src.services.mcp import user_environment as ue

pytestmark = pytest.mark.no_db

_DASHBOARD = "https://shapps.dataspace.copernicus.eu/dashboard/#/account/settings"

# servers/effis/environment.yaml
EFFIS = {
    "services": [
        {
            "id": "cdse",
            "label": "Copernicus Data Space",
            "provider": "Sentinel Hub",
            "url": "https://dataspace.copernicus.eu/",
            "tools": ["compute_metrics"],
            "credentials": [
                {
                    "header": "X-CDSE-Client-Id",
                    "label": "Copernicus client ID",
                    "help": "OAuth client id from the Sentinel Hub dashboard.",
                    "url": _DASHBOARD,
                    "secret": False,
                },
                {
                    "header": "X-CDSE-Client-Secret",
                    "label": "Copernicus client secret",
                    "help": "OAuth client secret from the Sentinel Hub dashboard.",
                    "url": _DASHBOARD,
                    "secret": True,
                },
            ],
        }
    ]
}

# servers/cdse/environment.yaml
CDSE = {
    "services": [
        {
            "id": "cdse",
            "label": "Copernicus Data Space",
            "provider": "Sentinel Hub",
            "url": "https://dataspace.copernicus.eu/",
            "tools": [
                "fetch_sentinel1_grd",
                "fetch_sentinel2",
                "fetch_sentinel3_olci",
                "fetch_sentinel3_slstr",
                "fetch_sentinel3_syn_l2",
                "fetch_sentinel5p_l2",
                "fetch_landsat_ot_l1",
                "fetch_dem",
            ],
            "credentials": [
                {
                    "variable": "CDSE_CLIENT_ID",
                    "label": "Copernicus client ID",
                    "help": "OAuth client id from the Sentinel Hub dashboard.",
                    "url": _DASHBOARD,
                    "secret": False,
                },
                {
                    "variable": "CDSE_CLIENT_SECRET",
                    "label": "Copernicus client secret",
                    "help": "OAuth client secret from the Sentinel Hub dashboard.",
                    "url": _DASHBOARD,
                    "secret": True,
                },
            ],
        }
    ]
}

_EXPLICIT = {
    "fields": [
        {"name": "client_id", "label": "Client ID", "header": "X-Client-Id", "secret": False}
    ]
}


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)


@pytest.fixture(autouse=True)
def _clear_warnings():
    ue._warned.clear()
    yield
    ue._warned.clear()


def _server(**kwargs) -> MCPServer:
    return MCPServer(
        name=kwargs.pop("name", "effis"),
        config=ToolConfig(url="https://example.com/mcp", transport=ToolTransport.STREAMABLE_HTTP),
        **kwargs,
    )


def test_effis_headers_fill_the_form_and_hide_the_header_names(flag_on):
    server = _server(user_environment=EFFIS)
    accounts = ue.public_accounts(server)
    assert accounts is not None
    [account] = accounts
    assert account.label == "Copernicus Data Space"
    assert account.provider == "Sentinel Hub"
    assert account.url == "https://dataspace.copernicus.eu/"
    assert account.tools == ["compute_metrics"]
    assert [item.model_dump() for item in account.credentials] == [
        {
            "name": "x_cdse_client_id",
            "label": "Copernicus client ID",
            "secret": False,
            "help_text": "OAuth client id from the Sentinel Hub dashboard.",
            "url": _DASHBOARD,
            "via": "header",
        },
        {
            "name": "x_cdse_client_secret",
            "label": "Copernicus client secret",
            "secret": True,
            "help_text": "OAuth client secret from the Sentinel Hub dashboard.",
            "url": _DASHBOARD,
            "via": "header",
        },
    ]
    rendered = account.model_dump_json()
    assert "X-CDSE-Client-Id" not in rendered
    assert "X-CDSE-Client-Secret" not in rendered

    spec = uc.effective_spec(server)
    assert spec is not None
    assert [field.header for field in spec.fields] == [
        "X-CDSE-Client-Id",
        "X-CDSE-Client-Secret",
    ]
    assert [field.name for field in spec.fields] == [
        "x_cdse_client_id",
        "x_cdse_client_secret",
    ]
    assert spec.help_url == _DASHBOARD
    assert spec.required is False
    assert spec.check is None


def test_cdse_variables_are_shown_and_do_not_become_a_form(flag_on):
    server = _server(name="cdse", user_environment=CDSE)
    accounts = ue.public_accounts(server)
    assert accounts is not None
    [account] = accounts
    assert account.tools[0] == "fetch_sentinel1_grd"
    assert account.tools[-1] == "fetch_dem"
    assert [item.via for item in account.credentials] == ["variable", "variable"]
    assert [item.name for item in account.credentials] == [
        "cdse_client_id",
        "cdse_client_secret",
    ]
    rendered = account.model_dump_json()
    assert "CDSE_CLIENT_ID" not in rendered
    assert "CDSE_CLIENT_SECRET" not in rendered
    assert uc.effective_spec(server) is None


def test_an_explicit_spec_wins_over_the_file(flag_on):
    server = _server(user_credentials=_EXPLICIT, user_environment=EFFIS)
    spec = uc.effective_spec(server)
    assert spec is not None
    assert [field.name for field in spec.fields] == ["client_id"]
    accounts = ue.public_accounts(server)
    assert accounts is not None
    assert accounts[0].credentials[0].name == "x_cdse_client_id"


def test_a_nested_account_keeps_its_place_and_its_header_joins_the_form(flag_on):
    raw = {
        "services": [
            {
                "id": "parent",
                "label": "Parent account",
                "credentials": [
                    {"header": "X-Parent-Key", "label": "Parent key", "secret": True}
                ],
                "services": [
                    {
                        "id": "child",
                        "label": "Child account",
                        "tools": ["child_tool"],
                        "credentials": [
                            {
                                "header": "X-Child-Key",
                                "label": "Child key",
                                "secret": False,
                                "url": "https://child.example.test/keys",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    accounts = ue.public_accounts(_server(user_environment=raw))
    assert accounts is not None
    assert accounts[0].services[0].id == "child"
    assert accounts[0].services[0].tools == ["child_tool"]
    spec = uc.effective_spec(_server(user_environment=raw))
    assert [field.name for field in spec.fields] == ["x_parent_key", "x_child_key"]
    assert spec.help_url == "https://child.example.test/keys"


def test_flag_off_and_a_user_owned_row_hide_the_file(flag_on, monkeypatch):
    server = _server(user_environment=EFFIS)
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", False)
    assert ue.public_accounts(server) is None
    assert uc.effective_spec(server) is None
    monkeypatch.setattr("src.config.FEATURE_TOOLKIT_CREDENTIALS", True)
    owned = _server(user_id="user-1", user_environment=EFFIS)
    assert ue.public_accounts(owned) is None
    assert uc.effective_spec(owned) is None


@pytest.mark.parametrize(
    "raw",
    [
        {"services": [{"id": "cdse", "label": "Copernicus", "note": "x", "credentials": []}]},
        {
            "services": [
                {
                    "id": "cdse",
                    "label": "Copernicus",
                    "credentials": [
                        {"header": "Authorization", "label": "Token", "secret": True}
                    ],
                }
            ]
        },
        {
            "services": [
                {
                    "id": "cdse",
                    "label": "Copernicus",
                    "credentials": [
                        {
                            "header": "X-Key",
                            "variable": "KEY",
                            "label": "Key",
                            "secret": True,
                        }
                    ],
                }
            ]
        },
        {
            "services": [
                {
                    "id": "cdse",
                    "label": "Copernicus",
                    "url": "javascript:alert(1)",
                    "credentials": [{"header": "X-Key", "label": "Key", "secret": True}],
                }
            ]
        },
        {
            "services": [
                {
                    "id": "one",
                    "label": "One",
                    "credentials": [{"header": "X-Key", "label": "Key", "secret": True}],
                },
                {
                    "id": "two",
                    "label": "Two",
                    "credentials": [{"header": "x-key", "label": "Key again", "secret": True}],
                },
            ]
        },
    ],
)
def test_a_broken_file_is_dropped(flag_on, raw):
    server = _server(user_environment=raw)
    assert ue.public_accounts(server) is None
    assert uc.effective_spec(server) is None


def test_a_broken_file_warns_once(flag_on, caplog):
    caplog.set_level(logging.WARNING)
    raw = {"services": []}
    server = _server(user_environment=raw)
    assert ue.public_accounts(server) is None
    assert uc.effective_spec(server) is None
    warnings = [r for r in caplog.records if "mcp.user_environment.invalid" in r.getMessage()]
    assert len(warnings) == 1
    assert "error=service" in warnings[0].getMessage()


def test_more_than_five_headers_stay_visible_but_do_not_open_a_form(flag_on, caplog):
    caplog.set_level(logging.WARNING)
    raw = {
        "services": [
            {
                "id": "wide",
                "label": "Wide account",
                "credentials": [
                    {"header": f"X-Key-{i}", "label": f"Key {i}", "secret": True}
                    for i in range(6)
                ],
            }
        ]
    }
    accounts = ue.public_accounts(_server(user_environment=raw))
    assert accounts is not None
    assert len(accounts[0].credentials) == 6
    assert uc.effective_spec(_server(name="wide", user_environment=raw)) is None
    assert "error=too_many_headers" in caplog.text
