# Changelog

Notable changes to the EVE backend. Newest first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Versions are tagged `vX.Y.Z`; the tag is
what promotes a build to staging, and production is promoted from there by an explicit dispatch.

## [Unreleased]

### Removed

- `sentence-transformers`, `transformers` and `langchain-huggingface` from `requirements.txt`. None
  of the three was ever imported: embeddings are computed by a remote OpenAI-compatible API through
  `EMBEDDING_URL`, not locally. They pulled in `torch`, which in its default PyPI form declares seven
  CUDA packages, on a service that runs on Fargate CPU and never sees a GPU.

  The image was 3.0 GB compressed, of which a single layer, the virtualenv, was 2,957 MB, and every
  Fargate task start downloads it from ECR before the process begins. It was paid on every deploy,
  every restart and every scale-out.

### Added

- The running version and commit are reported at `GET /api/health` and set on the FastAPI
  application, so an environment can be asked what it is serving instead of being assumed.

### Changed

- The ECS deploy step no longer keeps running for minutes after the service is already stable. The
  AWS SDK backs off exponentially up to 120 seconds between polls when left unconfigured; the
  interval is now pinned.

## [0.0.5] - 2026-08-02

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
