"""Diagrams for the backend docs. Run: python build_docs_diagrams.py"""

import sys
from pathlib import Path

sys.path.insert(0, "/Users/antoniolopez/.cursor/skills/diagram-generator/scripts")
from eve_svg import Canvas

ROOT = Path(__file__).resolve().parents[1]


def local_stack(path: Path) -> None:
    c = Canvas(
        title="Local stack",
        subtitle="Compose publishes the API on 8001. Qdrant and the models stay remote.",
    )
    browser = c.person(36, 118, "Browser")
    front = c.node(150, 104, "Frontend\n:5173", w=160, h=72)
    backend = c.node(390, 96, "Backend\n:8001", w=180, h=88)
    keycloak = c.node(150, 250, "Keycloak\n:8080", w=160, h=72)
    c.note_under(keycloak, "Realm eve")
    c.note(400, 36, "Listens on 8000 inside", w=200)

    data = c.cluster(650, 36, 400, 230, "Compose data", header=True)
    c.db(700, 100, "Mongo\n:27017")
    c.db(820, 100, "Redis\n:6379")
    c.db(940, 100, "MinIO\n:9000")
    c.note(748, 210, "Mailcatcher :6080", w=200)

    remote = c.cluster(650, 300, 400, 160, "Remote", header=True)
    c.db(720, 360, "Qdrant")
    c.node(860, 352, "LLM and\nembeddings", w=160, h=64)

    c.arrow(browser, "right", front, "left")
    c.arrow(front, "right", backend, "left", label="API")
    c.arrow(
        keycloak, "bottom", backend, "bottom", label="OIDC",
        via=[(230, 400), (480, 400)],
    )
    c.arrow(front, "bottom", keycloak, "top", label="sign-in")
    c.arrow(backend, "right", data, "left", label="stores")
    c.arrow(backend, "right", remote, "left", label="models", da=24)
    c.save(path, png=True)


def deployed(path: Path) -> None:
    c = Canvas(
        title="Deployed environments",
        subtitle="One platform. The private collection, the public filter, and the image differ.",
    )
    columns = [
        (24, "Dev", "dev.eve-chat.chat", "private-collections-dev", "prod, staging,\nor untagged", "Built on main", "agents main SHA"),
        (320, "Staging", "staging.eve-chat.chat", "private-collections-staging", "Same filter\nas dev", "Promoted digest", "agents.ref tag"),
        (616, "Prod", "eve-chat.chat", "private-collections", "prod, or\nuntagged", "Promoted digest", "agents.ref tag"),
    ]
    bottoms = []
    for x, name, host, private, public, image, agents in columns:
        c.node(x, 16, name, role="terminal", w=276, h=48)
        c.note(x + 18, 76, host, w=240)
        c.node(x, 128, f"Private vectors\n{private}", w=276, h=64)
        c.node(x, 212, f"Public points\n{public}", w=276, h=68)
        img = c.node(x, 300, f"{image}\n{agents}", w=276, h=64)
        bottoms.append(img)

    shared = c.cluster(24, 420, 868, 188, "Shared platform", header=True)
    labels = ["Cognito", "DocumentDB", "Redis", "S3", "AgentCore", "Qdrant", "Collector"]
    for i, label in enumerate(labels):
        c.node(44 + i * 122, 478, label, w=112, h=44)
    c.note(44, 542, "ECS behind CloudFront", w=200)
    c.note(260, 542, "One Qdrant cluster", w=180)
    c.note(456, 542, "Wiley skips the env filter", w=220)
    for i, node in enumerate(bottoms):
        c.arrow(node, "bottom", shared, "top", db=-300 + i * 300)
    c.save(path, png=True)


def _tree_spine(c: Canvas, root: object, children: list) -> None:
    """Elbow branches from the right of root to the left of each child."""
    spine = root.x + root.w + 28
    top = min(n.cy for n in children)
    bot = max(n.cy for n in children)
    c.polyline([(root.x + root.w, root.cy), (spine, root.cy)], head=False)
    if abs(bot - top) > 1:
        c.polyline([(spine, top), (spine, bot)], head=False)
    for n in children:
        c.polyline([(spine, root.cy), (spine, n.cy), (n.x, n.cy)], head=False)


def sandbox_files(path: Path) -> None:
    c = Canvas(
        title="Session tree",
        subtitle="The shell starts in workspace/. History is not a child of the session.",
    )
    rows = [
        (16, "workspace/", "highlight", "S3 workspace"),
        (100, "skills/", "default", "Session only"),
        (184, "artifacts/", "default", "S3 artifacts"),
        (268, "large results", "default", "S3, mirrored"),
    ]
    children = []
    stores = []
    for y, name, role, store in rows:
        children.append(c.node(250, y, name, role=role, w=190, h=60))
        stores.append(c.node(500, y, store, w=180, h=60))
    root = c.node(20, 128, "Session\nroot", role="terminal", w=150, h=72)
    _tree_spine(c, root, children)
    for src, dst in zip(children, stores):
        c.arrow(src, "right", dst, "left")
    off = c.node(20, 400, "Routed\noff", role="slate", w=150, h=64)
    hist = c.node(250, 404, "Chat history", role="slate", w=190, h=56)
    state = c.node(500, 404, "Graph state", w=180, h=56)
    _tree_spine(c, off, [hist])
    c.arrow(hist, "right", state, "left")
    c.note_under(hist, "/conversation_history/")
    c.save(path, png=True)


def memory_dump(path: Path) -> None:
    c = Canvas(
        title="Large tool result",
        subtitle="Over 20,000 tokens the full text leaves the thread.",
    )
    result = c.node(16, 24, "Tool result", role="terminal", w=160, h=72)
    gate = c.node(230, 16, "Over\n20,000?", role="decision", w=150, h=88)
    stay = c.node(230, 160, "Stays in\nthe thread", w=150, h=72)
    write = c.node(440, 24, "Write\nthe file", w=150, h=72)
    stub = c.node(650, 24, "Stub in\nthe thread", w=160, h=72)
    page = c.node(870, 24, "read_file\npages", role="terminal", w=150, h=72)
    c.arrow(result, "right", gate, "left")
    c.arrow(gate, "bottom", stay, "top", label="no")
    c.arrow(gate, "right", write, "left", label="yes")
    c.arrow(write, "right", stub, "left")
    c.arrow(stub, "right", page, "left")
    c.note_under(write, "/large_tool_results/")
    c.note_under(stub, "Path and 5 lines")
    c.save(path, png=True)


def memory_history(path: Path) -> None:
    c = Canvas(
        title="History tools",
        subtitle="Stored turns. The text that comes back is data, not instructions.",
    )
    store = c.node(16, 86, "messages", role="terminal", w=160, h=80)
    search = c.node(260, 16, "search_history", w=190, h=64)
    read = c.node(260, 176, "read_history", w=190, h=64)
    hits = c.node(520, 16, "Hits", w=150, h=64)
    raw = c.node(520, 176, "Raw messages", role="terminal", w=180, h=64)
    c.arrow(store, "right", search, "left", da=-16)
    c.arrow(store, "right", read, "left", da=16)
    c.arrow(search, "right", hits, "left")
    c.arrow(read, "right", raw, "left")
    c.note_under(search, "5 hits, at most 20")
    c.note_under(read, "5 on each side")
    c.note_under(raw, "Cut at 12,000 characters")
    c.save(path, png=True)


def memory_stores(path: Path) -> None:
    c = Canvas(
        title="Where a turn is kept",
        subtitle="Raw messages stay. Only the prompt drops them.",
    )
    rows = [
        (16, "messages", "Until delete"),
        (100, "checkpoint", "30 days"),
        (184, "summary copy", "Next summary"),
        (268, "S3 large results", "Per conversation"),
    ]
    children = [c.node(250, y, name, w=200, h=60) for y, name, _ in rows]
    notes = [detail for _, _, detail in rows]
    root = c.node(20, 128, "One\nconversation", role="terminal", w=160, h=72)
    _tree_spine(c, root, children)
    for n, detail in zip(children, notes):
        tw = c.text_w(detail, c.fs - 1) + 34
        c.note(n.x + n.w + 16, n.y + 12, detail, w=tw)
    c.save(path, png=True)


def sandbox_life(path: Path) -> None:
    c = Canvas(
        title="Sandbox lifetime",
        subtitle="One sandbox per conversation. The next message reuses it.",
    )
    turn = c.node(20, 36, "Turn\nstarts", role="terminal", w=130, h=72)
    lock = c.node(190, 36, "Lock", w=110, h=72)
    reuse = c.node(340, 28, "Reuse?", role="decision", w=140, h=88)
    attach = c.node(530, 36, "Attach", w=130, h=72)
    run = c.node(710, 36, "Turn\nruns", w=140, h=72)
    release = c.node(890, 36, "Release", w=130, h=72)
    idle = c.node(890, 180, "Idle\n600s", w=130, h=72)
    stop = c.node(890, 320, "Reaper\nstops", role="terminal", w=130, h=72)
    create = c.node(340, 180, "Create", w=140, h=72)
    restore = c.node(530, 180, "Restore", w=130, h=72)
    c.label(20, 308, "Other exits", anchor="start")
    discard = c.node(20, 328, "Delete\ndiscards", role="terminal", w=150, h=72)

    c.arrow(turn, "right", lock, "left")
    c.arrow(lock, "right", reuse, "left")
    c.arrow(reuse, "right", attach, "left", label="yes")
    c.arrow(attach, "right", run, "left", db=-16)
    c.arrow(run, "right", release, "left")
    c.arrow(release, "bottom", idle, "top")
    c.arrow(idle, "bottom", stop, "top")
    c.arrow(reuse, "bottom", create, "top", label="no")
    c.arrow(create, "right", restore, "left")
    c.arrow(restore, "top", run, "bottom", label="then run")
    c.note_under(create, "Gone, new provider, or AgentCore margin")
    c.note(700, 196, "Lock is not held during the turn", w=175)
    alone = c.node(220, 328, "No conversation", w=170, h=72)
    c.note_under(alone, "Destroyed when the turn ends")
    c.save(path, png=True)


def skills_load(path: Path) -> None:
    c = Canvas(
        title="Skills at turn start",
        subtitle="The names on this request are loaded. Packaged skills are always added.",
    )
    req = c.node(16, 24, "Selected\nin the request", role="highlight", w=190, h=80)
    look = c.node(270, 24, "Mongo\nlookup", w=160, h=80)
    merge = c.node(490, 24, "Merge", w=150, h=80)
    tree = c.node(700, 24, "Skill tree", w=160, h=80)
    sand = c.node(920, 24, "/skills/", role="terminal", w=150, h=80)
    c.arrow(req, "right", look, "left", label="turn start")
    c.arrow(look, "right", merge, "left")
    c.arrow(merge, "right", tree, "left")
    c.arrow(tree, "right", sand, "left")
    c.note_under(req, "Only these names")
    c.note_under(look, "Yours or shared")
    c.note_under(merge, "Packaged skill wins")
    c.note_under(tree, "Cached by digest")
    c.note_under(sand, "Skip if bytes match")
    empty = c.node(16, 210, "No names\non the request", role="decision", w=190, h=72)
    pack = c.node(270, 210, "Packaged\nskills only", w=160, h=72)
    c.arrow(empty, "right", pack, "left")
    c.save(path, png=True)


def prompt(path: Path) -> None:
    c = Canvas(
        title="How the prompt is built",
        subtitle="One system message, then the messages kept for this call.",
    )
    parts = [
        (0, "Clock\nevery call"),
        (180, "Pinned context\nnever rewritten"),
        (360, "EVE prompt\nreact prompts"),
        (540, "Deep addendum\nfiles and sandbox"),
    ]
    nodes = []
    for x, text in parts:
        nodes.append(c.node(x, 16, text, w=168, h=72))
    system = c.node(0, 150, "System message", role="terminal", w=708, h=52)
    for node in nodes:
        c.arrow(node, "bottom", system, "top")
    summary = c.node(0, 260, "Summary message", role="highlight", w=460, h=52)
    tail = c.node(0, 360, "Kept tail", w=460, h=52)
    new = c.node(0, 460, "New message", role="terminal", w=460, h=52)
    c.arrow(system, "bottom", summary, "top")
    c.arrow(summary, "bottom", tail, "top")
    c.arrow(tail, "bottom", new, "top")
    c.note(500, 268, "only after a summary", w=190)
    c.note(500, 368, "verbatim recent turns", w=200)
    c.note_under(new, "Attached files follow the query")
    tools = c.node(760, 150, "Tool schemas", role="slate", w=170, h=52)
    c.arrow(system, "right", tools, "left", label="beside")
    c.note_under(tools, "File rule is in the deep addendum")
    c.note(0, 560, "Other graphs: runner.py adds that rule on the first turn", w=480)
    c.save(path, png=True)


def main() -> None:
    local_stack(ROOT / "environments" / "local.svg")
    deployed(ROOT / "environments" / "deployed.svg")
    sandbox_files(ROOT / "sandbox" / "filesystem.svg")
    sandbox_life(ROOT / "sandbox" / "lifetime.svg")
    prompt(ROOT / "memory" / "prompt.svg")


if __name__ == "__main__":
    main()
