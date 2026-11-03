# Contributing to the EVE backend

## Pull requests

Branch from `main`, one concern per pull request. Merges are squashed: the pull request title follows Conventional Commits and becomes the changelog line, the pull request template says how. CI runs the tests on every pull request.

## Code review

Reviews follow [Conventional Comments](https://conventionalcomments.org/): every comment starts with a label, so the author sees at once what blocks the merge and what does not.

```
<label> [decorations]: <subject>

[discussion]
```

How a review reads:

- One short summary comment: what is good, then only the points that block the merge, one line each.
- Every other point inline, on the line it is about.
- Say why. When you are not sure, ask (`question`) instead of prescribing.
- When a point can wait, mark it `(non-blocking)`: the author picks this PR or the next one.

| Label | Use it for |
|---|---|
| `praise` | something done well; at least one per review, only when meant |
| `issue` | a concrete problem, paired with a fix |
| `suggestion` | an improvement: what and why |
| `question` | a possible concern you are not sure about |
| `todo` | a small change that is needed |
| `nitpick` | a matter of taste, always non-blocking |
| `thought` | an idea for later, non-blocking |
| `note` | something to be aware of, non-blocking |

Decorations: `(blocking)` must be resolved before merge, `(non-blocking)` must not hold the merge, `(if-minor)` only if the change turns out to be small.

```
issue (blocking): the cap is per file, so ten files can carry 6,000 chunks and the upload always hits the deadline. Could it be 600 per request?

suggestion (non-blocking): creating the collection after the slot reservation avoids an empty collection when the first upload is refused.

nitpick: a fixed message with a code here, like the other errors.
```
