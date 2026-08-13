# Changelog

Notable changes to the EVE backend. Newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

From v0.0.8 onward this file is written by [release-please](https://github.com/googleapis/release-please)
from the pull request titles merged into `main`, so entries are not added by hand. Work that has landed
but is not yet released is visible in the open `chore(main): release ...` pull request, which is also
what cuts the release: merging it commits the version, creates the `vX.Y.Z` tag, publishes the GitHub
Release and promotes staging. Production is promoted from there by an explicit dispatch.

## [0.0.10](https://github.com/eve-esa/backend/compare/v0.0.9...v0.0.10) (2026-08-13)


### Added

* encrypt custom-model keys with envelope encryption instead of per-key secrets ([#177](https://github.com/eve-esa/backend/issues/177)) ([6ac2515](https://github.com/eve-esa/backend/commit/6ac2515b6f72ef0a5dbe214f824fbb3240733d2c))
* the default EVE model answers via JSC where its picker entry is hidden ([#180](https://github.com/eve-esa/backend/issues/180)) ([8a3de20](https://github.com/eve-esa/backend/commit/8a3de20382551ccad1ba574d17798dd5d4bc483c))


### Fixed

* clear custom-model key material on delete ([#179](https://github.com/eve-esa/backend/issues/179)) ([d71f9cb](https://github.com/eve-esa/backend/commit/d71f9cb06df3ab2d2cd3cd2c9ff0082de0c654b2))

## [0.0.9](https://github.com/eve-esa/backend/compare/v0.0.8...v0.0.9) (2026-08-12)


### Added

* EVE answers through an ordered endpoint chain with failover ([#162](https://github.com/eve-esa/backend/issues/162)) ([4ef24d3](https://github.com/eve-esa/backend/commit/4ef24d3821269941a4839c3163a12c689c9a85b8))
* gate JSC platform model behind FEATURE_JSC_MODEL ([#159](https://github.com/eve-esa/backend/issues/159)) ([717afe7](https://github.com/eve-esa/backend/commit/717afe71b48a16a13f817adf85eed7cb15622454))
* structured tool fields on the agentic stream events ([#165](https://github.com/eve-esa/backend/issues/165)) ([428d933](https://github.com/eve-esa/backend/commit/428d9330156a7df02753c5e4237c294106227ce8))


### Fixed

* **deps:** clear the audited dependency vulnerabilities and automate updates ([#167](https://github.com/eve-esa/backend/issues/167)) ([13aeeec](https://github.com/eve-esa/backend/commit/13aeeecf614c376b26e7ae40fc3b7d83ed97d288))
* hold agentic generation until the stream subscriber attaches ([#164](https://github.com/eve-esa/backend/issues/164)) ([2d580e6](https://github.com/eve-esa/backend/commit/2d580e6fd6a232efc3686dabef04d72f4149c96a))
* inject the artifact instruction only on the first agentic turn ([#160](https://github.com/eve-esa/backend/issues/160)) ([d813156](https://github.com/eve-esa/backend/commit/d81315679e780bf02bacc95eb00d15eecd83ec25))
* open the agentic stream with a thinking notice ([#163](https://github.com/eve-esa/backend/issues/163)) ([e9fcbb2](https://github.com/eve-esa/backend/commit/e9fcbb2ed2e5f0c4681704daddd407126935264f))
* persist endpoint failures and keep retries on their pipeline ([#161](https://github.com/eve-esa/backend/issues/161)) ([fd1254f](https://github.com/eve-esa/backend/commit/fd1254fd8dbb53490d8266fe1dd2bf3d441b9f86))
* stop capping agentic answers with the endpoint probe budget ([#166](https://github.com/eve-esa/backend/issues/166)) ([dd0c7f6](https://github.com/eve-esa/backend/commit/dd0c7f69fcecfaffefe48177ee13644b51ece18b))
* unpack artifact_ids in message retry and keep SSE streams alive ([#157](https://github.com/eve-esa/backend/issues/157)) ([2065905](https://github.com/eve-esa/backend/commit/20659053f7ccd91e1e83263ded50d4b4bb6c37ad))

## [0.0.8](https://github.com/eve-esa/backend/compare/v0.0.7...v0.0.8) (2026-08-09)

### Added

* Feature flags are named for the features they control, each environment reports its own identity, and the OpenAI proxy no longer forwards the caller's token upstream ([#149](https://github.com/eve-esa/backend/pull/149))

### Removed

* `DELETE /artifacts/{id}`. An uploaded artifact is permanent ([#150](https://github.com/eve-esa/backend/pull/150))

### Fixed

* **ci:** report a missing dev image instead of dying on a raw AWS error ([#152](https://github.com/eve-esa/backend/issues/152)) ([207b0e4](https://github.com/eve-esa/backend/commit/207b0e44122f5a051434bc10e345e84cc202d105))
* **ci:** stop reporting success when a deploy started no tasks ([#153](https://github.com/eve-esa/backend/issues/153)) ([c27bf59](https://github.com/eve-esa/backend/commit/c27bf59b11c2227958ab36c38facbb779ae3bd62))

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
