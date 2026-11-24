"""Many dropped-bracket examples for the Mistral streaming repair.

SGLang removes every ``]`` when Mistral Large streams with tools
(sgl-project/sglang#44219). Each case drops that bracket from a real link,
or is prose that must stay. ``_close_link_labels`` is the pass under test.
"""

import logging

import pytest

from src.services.agents.core.runner import (
    _close_link_labels,
    rewrite_answer_file_links,
)

pytestmark = pytest.mark.no_db

_BANGS = ("", "!")
_TITLES = ("", ' "title"', ' "MCP: effis/compute_metrics"')
_LABELS = (
    "ndvi.png",
    "guide",
    "Chart",
    "a",
    "Sentinel-1 GRD",
    "plot_1",
    "NDVI",
    "图像",
    "name, with comma",
    "Band 4",
    "Q3 2026",
)
_PAREN_LABELS = (
    "Sentinel-2 BEFORE (s2_pre.jpg)",
    "Pre-flood SAR (Oct 2026)",
    "Name (a)",
    "Plot (final)",
    "Band (B04) crop",
    "Q (1) (2)",
    "see (note)",
    "chart (v1.2)",
    "Name (only",
    "River (km²)",
)
_TARGETS = (
    "/workspace/ndvi.png",
    "/workspace/flood/s2_pre.jpg",
    "/workspace/flood_indonesia_rgb/s2_post.jpg",
    "/artifacts/6afec33639a0d773c45aa7c1/severity_map.png",
    "/artifacts/6b044ad3b21a904b8785b71b/sentinel.jpg",
    "/conversations/c1/workspace/ndvi.png",
    "/skills/ndvi-map/SKILL.md",
    "/large_tool_results/call-1/out.txt",
    "./ndvi.png",
    "../outputs/ndvi.png",
    "https://example.com",
    "https://example.com/a/b.png",
    "https://example.com/a/b?q=1&x=2#top",
    "https://example.com:8443/img.jpg",
    "http://cdn.example/img.jpg",
    "https://onlinelibrary.wiley.com/cms/asset/1753318x/jfr370201-fig-0009-m.jpg",
    "//cdn.example/a.png",
)
_WRAPPERS = (
    "{link}",
    "- {link}",
    "  - {link}",
    "See {link} here.",
    "**{link}**",
    "1. {link}",
    "| {link} | after |",
    "Before. {link}\nNext line stays.",
)


def _cases():
    """``(broken, expected)`` pairs. Expected is what the closer must return."""
    pairs = []

    def add(broken, expected):
        pairs.append((broken, expected))

    for bang in _BANGS:
        for label in _LABELS + _PAREN_LABELS:
            for target in _TARGETS:
                for title in _TITLES:
                    broken = f"{bang}[{label}({target}{title})"
                    expect = f"{bang}[{label}]({target}{title})"
                    add(broken, expect)
                    for wrapper in _WRAPPERS:
                        add(wrapper.format(link=broken), wrapper.format(link=expect))

    # Bare file names. An image always closes. A plain link closes when the
    # target is a file name, or its last segment is the label.
    for label, target, title in (
        ("ndvi.png", "ndvi.png", ""),
        ("ndvi.png", "ndvi.png", ' "crop"'),
        ("guide", "docs/guide", ""),
    ):
        add(f"![{label}({target}{title})", f"![{label}]({target}{title})")
        add(f"[{label}({target}{title})", f"[{label}]({target}{title})")
    add("![plot(plots/plot.png)", "![plot](plots/plot.png)")
    add("[plot(plots/plot.png)", "[plot](plots/plot.png)")
    add("![click here(ndvi.png)", "![click here](ndvi.png)")
    add("[click here(ndvi.png)", "[click here](ndvi.png)")
    add("![Plot (final)(ndvi.png)", "![Plot (final)](ndvi.png)")
    add("![chart (v1.2)(plot.png)", "![chart (v1.2)](plot.png)")
    add("[Plot (final)(ndvi.png)", "[Plot (final)](ndvi.png)")
    add("[chart (v1.2)(plot.png)", "[chart (v1.2)](plot.png)")

    # Already closed, including a later citation on the same line.
    for whole in (
        "[guide](https://example.com)",
        "![chart](https://example.com/c.png)",
        "![Sentinel-2 BEFORE (s2_pre.jpg)](/workspace/flood/s2_pre.jpg)",
        "![Pre-flood SAR (Oct 2026)](/workspace/flood/pre.jpg)",
        '[pic.png](/artifacts/6afec33639a0d773c45aa7c1 "MCP: effis")',
        "See [the guide](https://example.com) and ![ndvi](ndvi.png).",
        "![a](/workspace/a.png) ![b](/workspace/b.png)",
        "Note [1] and [2].",
        "Use items[0] and items[i + 1].",
        "As in [see note (a) above.",
        "As in [see note (a) and (b) below.",
        "Ratio [x (y) per pixel.",
        "Wow![not a link",
        "No bracket here.",
        "(Oct 2026) alone",
        "[1]",
        "[S1]",
        "[text][ref]",
        "\\[ \\text{DI} = a - b",
        "sentinel-1-grd_2026-10-10_2026-10-20_6b1de08b.jpg",
    ):
        add(whole, whole)

    # Every ] removed from a short answer, which is what the stream delivers.
    stripped = (
        "See [the guide(https://example.com/guide) and ![ndvi(ndvi.png).\n"
        "Note [1 stays a citation.\n"
        "![Sentinel-2 BEFORE (s2_pre.jpg)(/workspace/flood/s2_pre.jpg)\n"
        "- [Sentinel-2 AFTER (s2_post.jpg)(/workspace/flood/s2_post.jpg)\n"
        "![Sentinel-3 OLCI before and after flood(/workspace/flood/s3.png)\n"
        "- [Pre-flood SAR (Oct 2026)(/workspace/flood/pre_flood_sar.jpg)\n"
        "Run `![x.png(x.png)` as an example.\n"
    )
    restored = (
        "See [the guide](https://example.com/guide) and ![ndvi](ndvi.png).\n"
        "Note [1 stays a citation.\n"
        "![Sentinel-2 BEFORE (s2_pre.jpg)](/workspace/flood/s2_pre.jpg)\n"
        "- [Sentinel-2 AFTER (s2_post.jpg)](/workspace/flood/s2_post.jpg)\n"
        "![Sentinel-3 OLCI before and after flood](/workspace/flood/s3.png)\n"
        "- [Pre-flood SAR (Oct 2026)](/workspace/flood/pre_flood_sar.jpg)\n"
        "Run `![x.png](x.png)` as an example.\n"
    )
    add(stripped, restored)

    # A label past the cap is left alone. One at the cap is restored.
    cap = "n" * 200
    add(f"[{cap}(/workspace/n.png)", f"[{cap}](/workspace/n.png)")
    over = "n" * 201
    add(f"[{over}(/workspace/n.png)", f"[{over}(/workspace/n.png)")

    # Two broken links on one line, then a closed image on the next.
    add(
        "![left(./a.png) and [right(https://example.com/b)\n![done](/workspace/c.png)",
        "![left](./a.png) and [right](https://example.com/b)\n![done](/workspace/c.png)",
    )
    # Empty image label, the ``]`` of ``![](path)`` having been dropped.
    add("![(/workspace/a.png)", "![](/workspace/a.png)")
    add("(/workspace/a.png)", "(/workspace/a.png)")
    return pairs


_PAIRS = _cases()


def test_thousands_of_dropped_brackets_round_trip():
    log = logging.getLogger("src.services.agents.core.runner")
    level = log.level
    log.setLevel(logging.WARNING)
    failures = []
    for index, (broken, expected) in enumerate(_PAIRS):
        got = _close_link_labels(broken)
        again = _close_link_labels(got)
        if got != expected or again != got:
            failures.append(f"{index}: {broken!r}\n  got {got!r}\n  want {expected!r}")
            if len(failures) >= 12:
                break
    try:
        assert not failures, f"{len(_PAIRS)} examples\n" + "\n".join(failures)
        assert len(_PAIRS) >= 4000
    finally:
        log.setLevel(level)


@pytest.mark.asyncio
async def test_a_streamed_answer_keeps_code_and_restores_the_links():
    """The full rewrite blanks code first, then restores the other links."""
    written = (
        "See [the guide(https://example.com/guide) and ![ndvi(ndvi.png).\n"
        "Note [1 stays a citation.\n"
        "![Sentinel-2 BEFORE (s2_pre.jpg)(/workspace/flood/s2_pre.jpg)\n"
        "- [Pre-flood SAR (Oct 2026)(/workspace/flood/pre_flood_sar.jpg)\n"
        "Run `![x.png(x.png)` as an example.\n"
        "```\n![y.png(y.png)\n```\n"
    )

    assert await rewrite_answer_file_links(written) == (
        "See [the guide](https://example.com/guide) and ![ndvi](ndvi.png).\n"
        "Note [1 stays a citation.\n"
        "![Sentinel-2 BEFORE (s2_pre.jpg)](/workspace/flood/s2_pre.jpg)\n"
        "- [Pre-flood SAR (Oct 2026)](/workspace/flood/pre_flood_sar.jpg)\n"
        "Run `![x.png(x.png)` as an example.\n"
        "```\n![y.png(y.png)\n```\n"
    )
