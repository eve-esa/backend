# AG-UI 1.0 schema

`ag-ui-1.0.schema.json` is the AG-UI protocol 1.0 JSON schema (`https://ag-ui.com/spec/1.0/schema.json`, published at https://docs.ag-ui.com, fetched 2026-11-12) with every prose `description` string removed. Descriptions take no part in validation; every other keyword is unchanged.

- Source file sha256: `4b5c93226838a0e72d88e6c5df20633c686c49fcb75d9be2815c6cbf9e48e71a`
- License: MIT, AG-UI protocol by CopilotKit, text in `LICENSE.ag-ui`.

Regenerate from a new schema file by loading it, dropping every `description` key whose value is a string, and writing it back with `json.dumps(..., indent=1)`. Bump it together with the `ag-ui-protocol` pin in `requirements-dev.txt`.
