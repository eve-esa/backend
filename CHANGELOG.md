# Changelog

Notable changes to the EVE backend. Newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v0.0.8 onward this file is written by [release-please](https://github.com/googleapis/release-please)
from the pull request titles merged into `main`, so entries are not added by hand. Work that has landed
but is not yet released is visible in the open `chore(main): release ...` pull request, which is also
what cuts the release: merging it commits the version, creates the `vX.Y.Z` tag, publishes the GitHub
Release and promotes staging. Production is promoted from there by an explicit dispatch.

## [0.0.7] - 2026-08-07

### Added

- Agentic answers: `POST /conversations/{id}/generate-agentic` and `/stream-generate-agentic`. The
  assistant can plan and call external tools mid-conversation instead of answering in one shot, with a
  LangGraph react runner streaming tool calls and results over SSE.
- `/mcp-servers` CRUD, a catalog of MCP toolkits that are enabled or disabled individually and reached
  over HTTPS on Bedrock AgentCore with a bearer token.

### Changed

- `GET /mcp-servers/{id}` returns `tools_error`, so a toolkit that fails to answer is reported as
  failing rather than shown as an empty tool list. Previously "this server exposes nothing" and "we
  could not reach it" produced byte-identical responses, which made a wall of unreachable toolkits
  indistinguishable from a wall of empty ones.

Which toolkits exist is configuration per environment, not code. Production deliberately has no agentic
configuration.

## [0.0.6] - 2026-08-05

### Removed

- `sentence-transformers`, `transformers` and `langchain-huggingface` from `requirements.txt`. None
  of the three was ever imported: embeddings are computed by a remote OpenAI-compatible API through
  `EMBEDDING_URL`, not locally. They pulled in `torch`, which in its default PyPI form declares seven
  CUDA packages, on a service that runs on Fargate CPU and never sees a GPU.

  The image was 3.0 GB compressed, of which a single layer, the virtualenv, was 2,957 MB, and every
  Fargate task start downloads it from ECR before the process begins. It was paid on every deploy,
  every restart and every scale-out.

### Fixed

- URL-encode the account activation link. Plus-addressed emails (`user+tag@domain`) decoded to a space
  in the browser, so `/verify` answered "user not found". Affected both the activation and the
  resend-activation mails.

### Changed

- The ECS deploy step no longer keeps running for minutes after the service is already stable. The
  AWS SDK backs off exponentially up to 120 seconds between polls when left unconfigured; the
  interval is now pinned.
- Artifact storage: S3-backed user and MCP artifacts, a capture interceptor and the `/artifacts` API.
- Image rendering support: demo image catalog, answer-side artifact stubs, `artifact_ids` over SSE, and
  catalog image URLs rewritten at the `generate_answer` boundary.
- User attachments are persisted on the agentic endpoints.

## [0.0.5] - 2026-08-02

### Added

- The running version and commit are reported at `GET /api/health` and set on the FastAPI
  application, so an environment can be asked what it is serving instead of being assumed.

First version that reports itself. Before this, no environment could be asked which build it was
running.

## [0.0.4] - 2026-07-31

Promotion drill with the environment source read inside the gated job.

## [0.0.3] - 2026-07-31

Single-approval promotion, and resilient index creation on a fresh DocumentDB.

## [0.0.2] - 2026-07-30

Promotion drill with the corrected pipeline: version tag normalisation and conditional health check.

## [0.0.1] - 2026-07-30

First promotion drill: validates tag-gated staging promotion by image digest.
