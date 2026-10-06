# Workflows

The CI and delivery graph for `eve-esa/backend`. Seven workflow files: five you can trigger,
one that only ever runs when another workflow calls it, and one that decides when a release
exists at all.

## The files

| File | Name | Triggered by | Kind |
| --- | --- | --- | --- |
| `ci.yml` | `check: ci` | `pull_request`, push to `main` | entry point |
| `pr-title.yml` | `check: pr title` | `pull_request` (opened, edited, reopened, synchronize) | entry point |
| `deploy-dev.yml` | `deploy: dev` | push to `main` (path filtered), `workflow_dispatch`, `workflow_call` from `release.yml` | entry point, also callable |
| `deploy-ecs.yml` | `internal: deploy to ecs` | `workflow_call` only | internal |
| `promote-staging.yml` | `promote: staging` | `workflow_dispatch` at a `v*` tag, dispatched by `release.yml` | entry point |
| `promote-prod.yml` | `promote: prod` | `workflow_dispatch` at a `v*` tag, one approval | entry point |
| `release.yml` | `release: draft or cut` | push to `main` | entry point |

## The graph

```
pull request
├── ci.yml                     "check: ci"          tests, mongo + redis services, no e2e
└── pr-title.yml               "check: pr title"    Conventional Commits on the PR title

push to main
├── ci.yml                     "check: ci"
├── deploy-dev.yml             "deploy: dev"        only when the path filter matches
│   ├── build                  builds and pushes sha-<short> and latest to the dev registry
│   └── deploy ──calls──> deploy-ecs.yml (environment: dev)
└── release.yml                "release: draft or cut"
    └── release-please         rewrites the open "chore(main): release x.y.z" PR

release PR merges (still release.yml, on the push it produces)
└── release-please             commits version.txt + CHANGELOG.md, tags, publishes the Release
    ├── deploy-dev ──calls──> deploy-dev.yml
    │   │                     the release commit is outside deploy-dev's path filter,
    │   │                     so nothing else would build it
    │   └── deploy ──calls──> deploy-ecs.yml (environment: dev)
    └── promote-staging        gh workflow run promote-staging.yml --ref <tag> -f version=<tag>
                               dispatched, not called: a call would run under refs/heads/main
                               and the esa-eve-staging environment only accepts v* refs
        │
        v
promote-staging.yml            "promote: staging"
├── resolve                    git rev-list the tag to its commit, take the short SHA
└── promote ──calls──> deploy-ecs.yml (environment: staging)
                               copies dev's sha-<short> image into the staging registry,
                               deploys it by digest, no rebuild

promote-prod.yml               "promote: prod"      manual dispatch at a v* tag, one approval
└── promote ──calls──> deploy-ecs.yml (environment: prod)
                               copies the staging image into the production registry,
                               deploys it by digest, no rebuild
```

One image is built, in dev, and the same digest travels to staging and to production. Nothing
downstream of `deploy-dev.yml` compiles anything.

## Why deploy-ecs shows no runs

`deploy-ecs.yml` is `workflow_call` only. There is no push, no schedule and no
`workflow_dispatch` on it, so it can never start a run of its own, and its page in the Actions
sidebar is permanently empty. Its `deploy` job appears inside the run of whichever workflow
called it: `deploy-dev.yml`, `promote-staging.yml` or `promote-prod.yml`.

An empty page is not a broken pipeline, and there is no way to remove it. GitHub has no setting
to hide a reusable workflow from the sidebar. It is an open feature request from 2022, still
unimplemented, and workflow subdirectories are unsupported: everything must sit flat in
`.github/workflows/`. Prefixing the `name:` with `internal:` is the only fix available, so
that is what the file does.

## Naming convention

Every workflow is named `Category: Detail`, dotnet/runtime style. The categories are:

- `check:` runs on a pull request, gates the merge
- `deploy:` builds and ships to an environment
- `internal:` reusable, `workflow_call` only, never run directly
- `promote:` moves an existing image to the next environment
- `release:` cuts versions and tags

The Actions sidebar sorts alphabetically and has no grouping of its own, so the prefix does the
grouping: the two checks sit together, the internals sit together, the promotions sit together.
Renaming a file is not free: `release.yml` dispatches the staging promotion by filename
(`promote-staging.yml`), and `deploy-dev.yml`'s path filter lists `deploy-dev.yml` and
`deploy-ecs.yml` by name.

## Job timeouts

Every job sets `timeout-minutes`: 15 normally, 30 for jobs that build an image or wait for ECS
to reach stability. The GitHub default is 360 minutes, which turns a hung step into six hours of
occupied runner. The 15 and 30 figures are community practice, not an official recommendation.
