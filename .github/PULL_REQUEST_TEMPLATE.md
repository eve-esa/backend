<!--
The title of this pull request becomes the commit message on main, because merges are squashed.
release-please reads those messages to write the changelog and pick the next version, so the title
has to follow Conventional Commits:

    <type>[(scope)][!]: <description>

    feat(artifacts): keep uploaded files permanently
    fix(auth): URL-encode the account activation link
    feat(api)!: drop DELETE /artifacts/{id}          <- ! marks a breaking change

Types: build chore ci docs feat fix perf refactor revert style test

Only feat, fix, perf, refactor and revert appear in the changelog. A CI check validates the title;
if it fails, edit the title in the browser and it re-runs on its own.

Do not edit CHANGELOG.md. It is generated.
-->

## What changed

## Why

## How it was verified
