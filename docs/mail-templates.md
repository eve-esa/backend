## Mail templates

Every mail the backend sends is rendered from a Jinja template. The copy can come from a published pack in S3, so a wording change does not need a release. Without a pack, or when the pack cannot be used, the backend renders the copy bundled in `src/templates/mail/`.

The loader is `src/services/mail_templates.py`. The mails, their default subjects and the variables each one receives are `MAIL_SPECS` in `src/services/account_notifications.py`.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `MAIL_TEMPLATES_BUCKET` | blank | Bucket that holds the pack. Blank: no S3 call, no background task, every mail from the bundled copy |
| `MAIL_TEMPLATES_REFRESH_S` | 300 | Seconds between two refreshes on each worker |

The task role needs `s3:GetObject` on the objects and `s3:ListBucket` on the bucket. `ListBucket` is there so a missing key answers 404 `NoSuchKey` and not 403.

### Pack layout

A pack is a flat set of objects at the bucket root: `manifest.yaml` and the files it lists. File names match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$`, so there are no folders. Publish every file first and `manifest.yaml` last: a reader that sees the new manifest then sees the new files. A reader that catches a publish in the middle sees a missing file or a hash mismatch and tries the same manifest again on its next refresh.

The hashes give integrity against a partial or out-of-order upload, not authenticity: whoever can write the bucket writes the manifest and its hashes too. The control is write access to the bucket, with object versioning as the audit trail and the rollback.

```yaml
contract: 1          # integer, must equal the backend's MAIL_PACK_CONTRACT
version: 1.4.0       # semantic version of the pack, shown in logs
files:               # every file of the pack and its sha256 (64 lowercase hex)
  base.html: 3f1c...
  account_pending.html: 9ab0...
  account_pending.txt: 77de...
mails:               # one entry per mail the pack carries
  account_pending:
    subject: Your EVE account is on hold for now
    html: account_pending.html
    text: account_pending.txt
```

- `subject` is plain text on one line, at most 200 characters. It is not a template. Control, format, private use and unassigned characters and the Unicode line and paragraph separators are refused (a tab, NUL, a vertical tab, U+2028, NEL). `send_cohort_invite --subject` follows the same rule.
- `html` and `text` must be listed in `files`.
- A pack may carry a subset of the mails. A mail it does not carry renders from the bundled copy.
- Mail names match `[a-z][a-z0-9_]{0,63}`.
- A mail the backend does not know is checked for syntax and otherwise ignored, so a pack can be published ahead of a backend that supplies it. A file that only such a mail lists but another file includes or extends is held to `email`, `app_url` and `logo_url`.
- Limits: `manifest.yaml` at most 1 MB, each file at most 512 KB.

### Mails and variables

Every mail receives `email` (the recipient address), `app_url` (the app for that environment) and `logo_url` (`<app_url>/branding/eve-logo.png`).

| Mail | Default subject | Extra variables |
| --- | --- | --- |
| `account_pending` | Your EVE account is on hold for now | none |
| `account_approved` | Your EVE account is ready | `after_hold` (bool: the account waited for approval) |
| `cohort_invite` | EVE is now open, come try it | `sign_in_url` (button target), `paragraphs` (list of strings, empty unless the command got `--text-file`) |
| `cohort_invite_signup` | Create your EVE account | `sign_up_url` (button target) |

A template may use only the variables of the mails that list it. A file no mail lists, such as the shared frame `base.html` or a partial, may use only `email`, `app_url` and `logo_url`. Every `extends` and `include` must name a file in the same pack, and no file may reach itself through them.

### What a template may contain

A pack is code that runs inside every backend worker, so the loader accepts only what a mail needs and refuses the rest before compiling anything:

- variables, text, `{% extends %}`, `{% block %}`, `{% include %}`, `{% if %}` and `{% elif %}`, comparisons, `and`, `or`, `not`, `~`, attribute and item access;
- `{% for %}` over a variable the mail receives (`paragraphs`, for example), one level deep, with no `for`, `include`, `extends` or `block` inside it;
- the filters `capitalize`, `count`, `d`, `default`, `e`, `escape`, `first`, `last`, `length`, `lower`, `string`, `striptags`, `title`, `trim`, `upper`, `urlencode`.

Arithmetic, function and method calls, `{% set %}`, `{% with %}`, macros, imports, `{% autoescape %}`, `|safe` and every other filter reject the pack with `reason=template`. The environment has no globals (`range`, `lipsum`, `cycler`, `joiner`, `namespace`, `dict` are gone).

Then the loader renders every mail the pack carries with sample values (each branch: with and without `after_hold`, with and without `paragraphs`). A render error rejects the pack with `reason=render`, a part above 256 KB with `reason=size`. All of this runs in the refresh thread, never on a send.

### Rendering

Templates run in Jinja's `ImmutableSandboxedEnvironment` with `StrictUndefined`. The html part always renders with autoescape on and the text part with it off, whatever the file names: an address is escaped in the HTML part and literal in the text part. `trim_blocks` and `lstrip_blocks` are on. A rendered part above 256 KB falls back to the bundled copy.

Links are not restricted: a pack can point a button or an image at any host, and the mail still leaves signed by our domain. Every link is reviewed in the pull request that changes the pack. A check of the rendered hosts against an allowlist is a follow-up.

The Jinja sandbox is the only barrier between a bucket writer and code in the task, so `jinja2` stays current: its security updates are never held.

`send_cohort_invite --subject` wins over the pack subject of `cohort_invite`. `cohort_invite_signup`, sent to a listed address with no account, always takes its subject from the pack or the bundled copy, and its button opens `--sign-up-url` (`FRONTEND_URL` otherwise).

### Contract

The contract number changes only when a pack written for the old one would break. A new mail or a new variable keeps contract 1. Removing or renaming a variable is contract 2. A pack that uses a variable the running backend does not supply is rejected, so publish such a pack only after every environment runs a backend that supplies it.

### Refresh

Each worker refreshes in the background. It sends `GetObject manifest.yaml` with `If-None-Match` set to the last ETag it read; a 304 ends the refresh. A changed manifest is parsed, each listed file is fetched by key and checked against its hash, the templates are validated, and the new pack replaces the old one in one step. A render that started before the swap finishes on the pack it started with. Startup waits for the first refresh for at most 5 seconds. `send_cohort_invite --apply` refreshes once before the first send, because a one-off task has no background refresh; a dry run makes no S3 call.

The S3 client is built once per process, on the event loop thread, from a boto3 session of its own. It uses a 2 second connect timeout, a 3 second read timeout and at most 2 attempts. A send never calls S3.

### Failures

| Case | Mail | Log |
| --- | --- | --- |
| Bucket blank | bundled | none |
| Bucket missing, access denied, timeout | last good pack, else bundled | `mail_templates_refresh_failed reason=`, then at most once an hour; `mail_template_fallback reason=no_pack` per mail while no pack is loaded |
| Manifest unchanged | current pack | none |
| Malformed YAML, schema error, wrong `contract`, size limit | last good pack | `mail_templates_rejected reason=yaml\|schema\|contract\|size` |
| Listed file missing, above 512 KB or hash mismatch (a publish in progress) | last good pack, retried on every refresh | `mail_templates_rejected reason=file\|size`, then at most once an hour |
| Syntax error, construct outside the allowlist, reference outside the pack or a cycle, unknown variable | last good pack | `mail_templates_rejected reason=template` |
| A mail fails or exceeds 256 KB in the trial render | last good pack | `mail_templates_rejected reason=render\|size` |
| Mail not in the pack | bundled for that mail | `mail_template_fallback reason=absent` |
| Error while rendering a real send (for example an undefined attribute for one address) | bundled for that mail, the mail is still sent | `mail_template_fallback reason=<error class>` |
| A rendered part above 256 KB, or a subject the message build refuses | bundled for that mail, the mail is still sent | `mail_template_fallback reason=size\|subject` |

A rejected manifest (YAML, schema, contract, manifest size, template, render) is reported once: the next refresh sends its ETag and gets a 304. Publish a fixed manifest to retry. A missing file or a hash mismatch is not pinned: the next refresh reads the same manifest again, so uploading the missing file heals it. A successful load logs `mail_templates_loaded version= etag=`. `account_mail_sent` and `cohort_invite_sent` carry `template_source=pack|bundled` and `pack_version=`. No line carries an address.

### Checking a pack before publishing

```
python -m src.commands.render_mail_pack --pack ./pack --out ./out
```

It runs the same validation as the backend, then renders every variant of `<pack>/fixtures.yaml` (or `--fixtures`) to `<out>/<variant>.html` and `.txt` and prints each subject. `--out` must be a new or empty directory. A symlink in the pack exits 1. The variant name is the file stem, so `04-on-hold` writes `04-on-hold.html`. The command reads only the pack directory: it needs no database, no network and no real settings. There is no fallback here: a rejected pack, a fixture whose `context` keys differ from the mail's variables, a mail the pack does not carry or a render error exits 1. `fixtures.yaml` is never published.

```yaml
variants:
  - name: account-pending
    mail: account_pending
    context:
      email: person@example.com
      app_url: https://app.example.com
      logo_url: https://app.example.com/branding/eve-logo.png
```

To fill in the hashes after editing templates:

```
python -m src.commands.render_mail_pack --pack ./pack --write-hashes
```

It sets the value of every entry under `files:` in `manifest.yaml` to the sha256 of that file and leaves every other line, comment and line ending as written. A listed file that is missing, a line under `files:` it cannot read (a quoted key, an anchor, a value with a comment glued to it) or a result whose parsed `files` differ from the computed digests exits 1 and leaves the manifest untouched; otherwise the manifest is replaced in one step. Add `--out` to validate and render in the same run.

### Rollback

Publish the previous pack, or copy the previous version of each object over the current one (`manifest.yaml` last). Workers pick it up within `MAIL_TEMPLATES_REFRESH_S`. To return to the bundled copy, blank `MAIL_TEMPLATES_BUCKET` and restart the service.
