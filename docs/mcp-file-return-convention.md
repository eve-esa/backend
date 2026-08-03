# MCP tool convention: returning files (the "effis" fix)

Audience: mcp-tool-registry maintainers (cc Jino). Scope: how MCP tools
should return generated files (images, reports, charts, ...) so that any MCP
client — not just EVE — can actually retrieve them.

## The problem

Several registry tools compute a result, write it to a file on the server
container's filesystem, and return only a path string or a bare text summary
(`effis`'s `compute_metrics`-style tools are the clearest example). That path
is meaningless to a client:

- It refers to a location inside the tool server's own container. The client
  process has no filesystem access to that container, and by the time the
  client could act on the path the container may already have cycled
  (ephemeral compute, autoscaling, redeploys).
- Nothing in the MCP result signals "this is a file, here's how to fetch it."
  A client has to know out-of-band that a given tool writes files and where,
  which doesn't scale across a registry of independently maintained tools.

The tempting alternative — base64-encode the file and inline it in the tool
result — doesn't scale either. Large inline payloads blow up context windows,
get duplicated across every hop that touches the message (LLM context,
checkpointer, SSE stream, logs), and several MCP SDKs choke on them; see
[modelcontextprotocol/python-sdk#2557](https://github.com/modelcontextprotocol/python-sdk/issues/2557)
for concrete failure modes with large inline base64 content.

## The convention

**Small, genuinely inline-appropriate content** (a small icon, a short
generated snippet) may be returned directly as `ImageContent` /
`EmbeddedResource` with the bytes inline. Keep this to content you'd be happy
to see duplicated verbatim into an LLM's context window — a rough ceiling is
a few tens of KB.

**Anything else — and in particular anything already being written to disk —
follows a three-step pattern:**

1. **Write** the file wherever the tool already writes it (temp dir, cache
   dir, whatever the tool's runtime uses today). No change needed here.
2. **Expose** that file as an MCP resource: register it (or a template
   parameterized by whatever varies per call, e.g. a region or job id) via
   `@mcp.resource(...)`, with an accurate `mime_type` and a URI whose last
   path segment is a sensible display filename (clients that persist the
   resource use that segment as the filename).
3. **Return** a `ResourceLink` block (`type="resource_link"`, `uri`, `name`,
   `mimeType`, `description`) pointing at that resource, alongside whatever
   plain-text summary the tool already returns. Do not return the file path
   as text anywhere in the result — the `ResourceLink` is the file's only
   identity as far as the result is concerned.

Naming/mime expectations:

- `mimeType` must reflect the actual content (`image/png`, `text/csv`,
  `application/pdf`, ...) — clients use it to decide how to handle/display
  the fetched bytes.
- The resource URI's final path segment doubles as the display filename once
  a client resolves it, so make it recognizable (`metrics-eu-south-1.png`,
  not a UUID or a hash).
- Size guidance: if the file is large enough that you were tempted to inline
  it as base64, it should be a `ResourceLink`, not `EmbeddedResource`. There's
  no hard registry-wide byte limit prescribed here since that's a per-client
  policy (see what EVE enforces below); a tool should not itself refuse to
  expose a resource for being "too big."

## What EVE guarantees client-side

EVE's `ArtifactInterceptor` (`src/services/mcp/artifact_ingestion.py`, on
`feat/artifact-storage`) sits in the tool-call interceptor chain and handles
both shapes of the convention above transparently:

- **Capture**: `ImageContent`/`EmbeddedResource` blocks are read directly from
  the tool result; `ResourceLink` blocks are resolved via
  `session.read_resource(uri)` under a bounded timeout
  (`ARTIFACT_RESOURCE_READ_TIMEOUT_S`), so a slow or unreachable resource
  fails open (the link is left untouched) rather than hanging the tool call.
- **S3 persistence**: captured bytes are written to per-user object storage
  and recorded as an `Artifact` document (owner, content type, size,
  conversation/message linkage), independent of the tool server's lifecycle.
- **Context protection**: the original block is stripped from what reaches
  the LLM/checkpointer/SSE stream and replaced with a small stub (a markdown
  link/image reference plus a one-line JSON blob with the artifact id and
  URL) — a guard against exactly the inline-base64 blowup described above,
  and against files large enough to matter even if a tool did return them
  inline. A size cap (`ARTIFACT_MAX_BYTES`) and a per-call count cap
  (`ARTIFACT_MAX_PER_TOOL_CALL`) apply to both paths.
- **Provenance**: every stub's markdown title attributes its origin as
  `"MCP: {server}/{tool}"` (e.g. `"MCP: dummy/compute_metrics"`), so a user
  looking at a chat transcript can trace an artifact back to the tool that
  produced it without inspecting logs.

## Reference implementation

`tests/e2e/dummy_mcp_server/server.py`'s `compute_metrics` tool is the
reference fix for effis-style tools, exercised end-to-end by
`tests/e2e/test_artifact_e2e.py::test_compute_metrics_effis_pattern_e2e`.
Diff-sized shape (elided):

```python
# 1. write the file, exactly as an effis-style tool already does today
png_bytes = make_solid_png(48, 48, rgb)
resource_filename = f"metrics-{region}.png"
with open(_metrics_png_path(resource_filename), "wb") as f:
    f.write(png_bytes)

# 2. expose it as a resource (template, since `region` varies per call)
@mcp.resource("resource://dummy/compute-metrics/{filename}", mime_type="image/png")
def compute_metrics_resource(filename: str) -> bytes:
    with open(_metrics_png_path(filename), "rb") as f:
        return f.read()

# 3. return a ResourceLink instead of a bare path/summary
return [
    TextContent(type="text", text=f"Computed metrics for region={region!r}: ..."),
    ResourceLink(
        type="resource_link",
        name=resource_filename,
        uri=f"resource://dummy/compute-metrics/{resource_filename}",
        mimeType="image/png",
        description=f"Metrics chart for region={region!r}, generated by compute_metrics",
    ),
]
```

No other change is required of the tool's compute logic — this is purely
about how the already-written file gets surfaced in the result.
