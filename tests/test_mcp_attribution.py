from src.services.mcp.attribution import attribution_from_citation

EFFIS_CITATION = {
    "cff-version": "1.2.0",
    "message": "If you use this MCP server, please cite it using the metadata from this file.",
    "type": "software",
    "title": "EFFIS Fire Detection MCP Server",
    "abstract": "MCP server for European wildfire analysis.",
    "authors": [
        {
            "given-names": "Antonio",
            "family-names": "Lopez",
            "alias": "Lopez-Antonio",
            "affiliation": "Pi School",
            "email": "antonio@example.com",
        }
    ],
    "license": "Apache-2.0",
    "repository-code": "https://github.com/eve-esa/mcp-tool-registry/tree/main/servers/effis",
}


def test_maps_the_registry_effis_citation():
    attribution = attribution_from_citation(EFFIS_CITATION, server_name="effis")

    assert attribution is not None
    assert attribution.title == "EFFIS Fire Detection MCP Server"
    assert attribution.license == "Apache-2.0"
    assert attribution.repository_url.endswith("/servers/effis")
    assert attribution.preferred_citation is None
    [author] = attribution.authors
    assert author.name == "Antonio Lopez"
    assert author.alias == "Lopez-Antonio"
    assert author.github_url == "https://github.com/Lopez-Antonio"
    assert author.affiliation == "Pi School"


def test_never_exposes_contact_details():
    attribution = attribution_from_citation(EFFIS_CITATION)

    assert "antonio@example.com" not in attribution.model_dump_json()


def test_entity_author_name_particle_and_non_handle_alias():
    attribution = attribution_from_citation(
        {
            "title": "Some server",
            "authors": [
                {"name": "European Space Agency", "website": "https://esa.int"},
                {
                    "given-names": "Ludwig",
                    "name-particle": "van",
                    "family-names": "Beethoven",
                    "alias": "not a handle",
                    "orcid": "https://orcid.org/0000-0002-1825-0097",
                },
            ],
        }
    )

    entity, person = attribution.authors
    assert entity.name == "European Space Agency"
    assert entity.website == "https://esa.int"
    assert person.name == "Ludwig van Beethoven"
    assert person.alias == "not a handle"
    assert person.github_url is None
    assert person.orcid == "https://orcid.org/0000-0002-1825-0097"


def test_preferred_citation_links_the_doi_and_takes_year_from_date():
    attribution = attribution_from_citation(
        {
            **EFFIS_CITATION,
            "license": ["Apache-2.0", "MIT"],
            "preferred-citation": {
                "type": "article",
                "title": "Burnt area mapping",
                "authors": [{"given-names": "Ada", "family-names": "Lovelace"}],
                "date-published": "2024-03-01",
                "doi": "10.1234/abcd",
                "url": "https://example.com/paper",
            },
        }
    )

    assert attribution.license == "Apache-2.0 OR MIT"
    ref = attribution.preferred_citation
    assert ref.title == "Burnt area mapping"
    assert ref.authors == ["Ada Lovelace"]
    assert ref.year == "2024"
    assert ref.url == "https://doi.org/10.1234/abcd"


def test_drops_links_that_are_not_http():
    attribution = attribution_from_citation(
        {
            "title": "Some server",
            "authors": [
                {
                    "name": "European Space Agency",
                    "website": "javascript:alert(1)",
                    "orcid": "javascript:alert(1)",
                }
            ],
            "repository-code": "javascript:alert(1)",
            "url": "https://example.com/fallback",
            "preferred-citation": {
                "title": "Not a paper",
                "doi": "javascript:alert(1)",
                "url": "javascript:alert(1)",
            },
        }
    )

    assert attribution.authors[0].website is None
    assert attribution.authors[0].orcid is None
    assert attribution.repository_url == "https://example.com/fallback"
    assert attribution.preferred_citation.url is None


def test_keeps_a_doi_that_is_already_a_url():
    attribution = attribution_from_citation(
        {
            "title": "Some server",
            "authors": [{"name": "European Space Agency"}],
            "preferred-citation": {
                "title": "Burnt area mapping",
                "doi": "https://doi.org/10.1234/abcd",
            },
        }
    )

    assert attribution.preferred_citation.url == "https://doi.org/10.1234/abcd"


def test_missing_or_malformed_citation_gives_none():
    assert attribution_from_citation(None) is None
    assert attribution_from_citation("cff-version: 1.2.0") is None
    assert attribution_from_citation({}) is None
    assert attribution_from_citation({"authors": "Antonio Lopez"}) is None
