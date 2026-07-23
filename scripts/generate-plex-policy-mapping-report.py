#!/usr/bin/env python3

import argparse
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPLICATION_REPORT = ROOT / "tests" / "policies" / "replication-report.json"
HOOK_MAPPING = ROOT / "tests" / "policies" / "policy-hook-mapping.json"
CATALOG = ROOT / "plex-serving-policy-wiki" / "catalog.json"
OUTPUT = ROOT / "plex_policy_mapping_report.html"
OPERATIONS = ("admit", "route", "schedule", "cache", "feedback")
KINDS = {"decision", "state", "declared"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the standalone PLEX policy-to-hook mapping report."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in HTML does not match generated output",
    )
    args = parser.parse_args()

    report = read_json(REPLICATION_REPORT)
    mapping = read_json(HOOK_MAPPING)
    catalog = read_json(CATALOG)
    entries = build_entries(report, mapping, catalog)
    rendered = render(report, mapping, entries)
    rendered = rendered.encode("ascii", "xmlcharrefreplace").decode("ascii")
    validate_rendered(rendered, entries)

    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != rendered:
            raise SystemExit(
                f"{OUTPUT.relative_to(ROOT)} is stale; run "
                "scripts/generate-plex-policy-mapping-report.py"
            )
        return
    OUTPUT.write_text(rendered)


def build_entries(report: dict, mapping: dict, catalog: list[dict]) -> list[dict]:
    if report["contract"] != mapping["contract"]:
        raise SystemExit("replication report and hook mapping contracts differ")
    if tuple(mapping["operations"]) != OPERATIONS:
        raise SystemExit(f"hook mapping operations must be {OPERATIONS}")

    catalog_by_replication = {}
    for paper in catalog:
        status = paper.get("plex_v0_6_replication")
        if status is None:
            continue
        replication_id = status["replication_id"]
        if replication_id in catalog_by_replication:
            raise SystemExit(f"duplicate catalog replication {replication_id}")
        catalog_by_replication[replication_id] = paper

    report_ids = [entry["id"] for entry in report["entries"]]
    mapping_ids = list(mapping["policies"])
    if set(report_ids) != set(mapping_ids):
        missing = sorted(set(report_ids) - set(mapping_ids))
        extra = sorted(set(mapping_ids) - set(report_ids))
        raise SystemExit(f"hook mapping mismatch: missing={missing}, extra={extra}")
    if set(report_ids) != set(catalog_by_replication):
        missing = sorted(set(report_ids) - set(catalog_by_replication))
        extra = sorted(set(catalog_by_replication) - set(report_ids))
        raise SystemExit(f"catalog replication mismatch: missing={missing}, extra={extra}")

    entries = []
    for index, entry in enumerate(report["entries"]):
        replication_id = entry["id"]
        hooks = mapping["policies"][replication_id]
        implements = set(entry["implements"])
        if set(hooks) != implements:
            raise SystemExit(
                f"{replication_id}: hook notes {sorted(hooks)} do not match "
                f"implements {sorted(implements)}"
            )
        for operation, hook in hooks.items():
            if operation not in OPERATIONS:
                raise SystemExit(f"{replication_id}: unknown operation {operation}")
            if hook.get("kind") not in KINDS:
                raise SystemExit(
                    f"{replication_id}/{operation}: unknown mapping kind "
                    f"{hook.get('kind')!r}"
                )
            if not hook.get("summary"):
                raise SystemExit(f"{replication_id}/{operation}: empty summary")

        paper = catalog_by_replication[replication_id]
        catalog_operations = set(paper["plex_v0_6_replication"]["operations"])
        if catalog_operations != implements:
            raise SystemExit(f"{replication_id}: catalog operation mismatch")

        entries.append(
            {
                **entry,
                "index": index,
                "hooks": hooks,
                "paper": paper,
                "hook_count": len(implements),
            }
        )

    counts = Counter(
        operation for entry in entries for operation in entry["implements"]
    )
    if dict(sorted(counts.items())) != report["operation_counts"]:
        raise SystemExit("operation counts do not match the replication report")
    return entries


def render(report: dict, mapping: dict, entries: list[dict]) -> str:
    total = len(entries)
    multi_hook = sum(entry["hook_count"] > 1 for entry in entries)
    declared_only = sum(
        hook["kind"] == "declared"
        for entry in entries
        for hook in entry["hooks"].values()
    )
    custom_feedback = report["operation_counts"].get("feedback", 0) - declared_only
    optional = Counter(
        mechanic
        for entry in entries
        for mechanic in entry.get("optional_mechanics", [])
    )
    categories = Counter(entry["paper"]["category"] for entry in entries)
    category_options = "\n".join(
        f'<option value="{escape(category)}">{escape(category)} ({count})</option>'
        for category, count in sorted(categories.items())
    )
    coverage_cards = "\n".join(
        coverage_card(
            operation,
            mapping["operations"][operation],
            report["operation_counts"].get(operation, 0),
            total,
        )
        for operation in OPERATIONS
    )
    filter_buttons = "\n".join(
        (
            f'<button class="filter-chip hook-{operation}" type="button" '
            f'data-hook-filter="{operation}" aria-pressed="false">'
            f'{escape(mapping["operations"][operation]["label"])} '
            f'<span>{report["operation_counts"].get(operation, 0)}</span></button>'
        )
        for operation in OPERATIONS
    )
    rows = "\n".join(render_entry(entry, mapping["operations"]) for entry in entries)
    mechanics = ", ".join(
        f"{escape(mechanic)} ({count})" for mechanic, count in sorted(optional.items())
    )
    operation_headers = "\n".join(
        (
            f'<th scope="col" class="hook-heading hook-{operation}">'
            f'<span>{escape(mapping["operations"][operation]["label"])}</span>'
            f'<small>{escape(mapping["operations"][operation]["authority"])}</small>'
            "</th>"
        )
        for operation in OPERATIONS
    )

    page = PAGE
    replacements = {
        "__TOTAL__": str(total),
        "__PASSING__": str(report["passing_count"]),
        "__MULTI_HOOK__": str(multi_hook),
        "__DECLARED_ONLY__": str(declared_only),
        "__CUSTOM_FEEDBACK__": str(custom_feedback),
        "__SCHEDULE_COUNT__": str(report["operation_counts"].get("schedule", 0)),
        "__ADMIT_COUNT__": str(report["operation_counts"].get("admit", 0)),
        "__CONTRACT__": (
            f'{report["contract"]["major"]}.{report["contract"]["minor"]}'
        ),
        "__COVERAGE_CARDS__": coverage_cards,
        "__FILTER_BUTTONS__": filter_buttons,
        "__CATEGORY_OPTIONS__": category_options,
        "__OPERATION_HEADERS__": operation_headers,
        "__ROWS__": rows,
        "__OPTIONAL_MECHANICS__": mechanics or "None",
    }
    for marker, value in replacements.items():
        page = page.replace(marker, value)
    return page


def coverage_card(operation: str, definition: dict, count: int, total: int) -> str:
    percent = round(count * 100 / total)
    return (
        f'<button class="coverage-card hook-{operation}" type="button" '
        f'data-hook-filter="{operation}" aria-pressed="false">'
        f'<span class="coverage-name">{escape(definition["label"])}</span>'
        f'<strong>{count}<small> / {total}</small></strong>'
        f'<span class="coverage-bar"><i style="width:{percent}%"></i></span>'
        f'<span class="coverage-authority">{escape(definition["authority"])}</span>'
        "</button>"
    )


def render_entry(entry: dict, definitions: dict) -> str:
    paper = entry["paper"]
    replication_id = entry["id"]
    title = entry["title"]
    label = paper["label"]
    source_url = checked_url(entry["source_url"])
    wiki_path = f"plex-serving-policy-wiki/papers/{paper['slug']}.md"
    implementation_path = f"tests/policies/paper-{replication_id}/src/lib.rs"
    metadata_path = (
        f"tests/policies/replications/{replication_id}/metadata.json"
    )
    case_path = entry["case"]
    expected_path = entry["expected"]
    artifact_url = paper.get("artifact_url")
    category = paper["category"]
    search_text = " ".join(
        [
            replication_id,
            label,
            title,
            category,
            paper.get("venue") or "",
            " ".join(entry["policy_kernel"]),
            " ".join(entry["required_facts"]),
            " ".join(entry.get("optional_mechanics", [])),
            " ".join(entry["deferred_mechanics"]),
        ]
    ).lower()
    hook_cells = "\n".join(
        render_hook_cell(operation, entry["hooks"].get(operation), definitions[operation])
        for operation in OPERATIONS
    )
    year = paper.get("year") or "n/a"
    venue = paper.get("venue") or entry["paper_version"]
    citation = paper.get("citation_count")
    citation_text = "not resolved" if citation is None else str(citation)
    authors = ", ".join(paper.get("authors", [])[:4])
    if len(paper.get("authors", [])) > 4:
        authors += ", et al."
    optional_mechanics = entry.get("optional_mechanics", [])
    deferred = entry["deferred_mechanics"]
    artifact_link = (
        f'<a href="{escape(checked_url(artifact_url))}" target="_blank" '
        'rel="noopener">Artifact</a>'
        if artifact_url
        else '<span class="muted-link">No confirmed artifact</span>'
    )

    return f"""
<tr class="policy-row" data-index="{entry['index']}" data-name="{escape(label.lower())}"
    data-year="{escape(year)}" data-hook-count="{entry['hook_count']}"
    data-ops="{' '.join(entry['implements'])}" data-category="{escape(category)}"
    data-search="{escape(search_text)}">
  <th scope="row" class="paper-cell">
    <button class="expand-button" type="button" aria-expanded="false"
            aria-controls="detail-{escape(replication_id)}"
            title="Show implementation evidence">+</button>
    <span class="paper-copy">
      <a class="paper-name" href="{escape(source_url)}" target="_blank"
         rel="noopener">{escape(label)}</a>
      <span class="paper-title">{escape(title)}</span>
      <span class="paper-meta">{escape(year)} &middot; {escape(venue)} &middot;
        <a href="{escape(wiki_path)}">PLEX wiki</a>
      </span>
    </span>
  </th>
  {hook_cells}
  <td class="evidence-cell">
    <span class="status-pass">passing</span>
    <strong>{entry['hook_count']} hook{"s" if entry['hook_count'] != 1 else ""}</strong>
    <small>policy-kernel reproduction</small>
  </td>
</tr>
<tr class="detail-row" id="detail-{escape(replication_id)}" hidden>
  <td colspan="7">
    <div class="detail-grid">
      <section>
        <h3>Policy kernel</h3>
        {render_list(entry['policy_kernel'])}
      </section>
      <section>
        <h3>Required facts</h3>
        {render_chips(entry['required_facts'], 'fact')}
        <h3>Optional mechanics</h3>
        {render_chips(optional_mechanics, 'mechanic', empty='None')}
      </section>
      <section>
        <h3>Deferred engine mechanics</h3>
        {render_list(deferred, empty='None')}
        <p class="evidence-note"><strong>Evidence:</strong>
          {escape(entry['evidence_level'])}. This reproduces the policy kernel,
          not the paper's complete serving system.</p>
      </section>
      <section>
        <h3>Paper and implementation</h3>
        <p>{escape(authors)}</p>
        <p>{escape(entry['paper_version'])}; catalog citations: {escape(citation_text)}.</p>
        <nav class="detail-links">
          <a href="{escape(source_url)}" target="_blank" rel="noopener">Paper</a>
          {artifact_link}
          <a href="{escape(wiki_path)}">Wiki</a>
          <a href="{escape(implementation_path)}">Component</a>
          <a href="{escape(metadata_path)}">Metadata</a>
          <a href="{escape(case_path)}">Case</a>
          <a href="{escape(expected_path)}">Expected trace</a>
        </nav>
      </section>
    </div>
  </td>
</tr>""".strip()


def render_hook_cell(operation: str, hook: dict | None, definition: dict) -> str:
    if hook is None:
        return (
            f'<td class="hook-cell hook-{operation} inactive" '
            f'aria-label="{escape(definition["label"])} not used"><span>--</span></td>'
        )
    kind = hook["kind"]
    label = {
        "decision": "decision",
        "state": "state update",
        "declared": "declared only",
    }[kind]
    return (
        f'<td class="hook-cell hook-{operation} active kind-{kind}">'
        f'<span class="mapping-kind">{escape(label)}</span>'
        f'<p>{escape(hook["summary"])}</p></td>'
    )


def render_list(values: list[str], empty: str = "None") -> str:
    if not values:
        return f'<p class="empty-value">{escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in values) + "</ul>"


def render_chips(values: list[str], kind: str, empty: str = "None") -> str:
    if not values:
        return f'<p class="empty-value">{escape(empty)}</p>'
    return '<div class="chips">' + "".join(
        f'<span class="chip chip-{escape(kind)}">{escape(value)}</span>'
        for value in values
    ) + "</div>"


def validate_rendered(rendered: str, entries: list[dict]) -> None:
    if rendered.count('class="policy-row"') != len(entries):
        raise SystemExit("generated HTML policy row count is incorrect")
    if rendered.count('class="detail-row"') != len(entries):
        raise SystemExit("generated HTML detail row count is incorrect")
    for entry in entries:
        if f'data-search="{escape(entry["id"])}' not in rendered:
            raise SystemExit(f"generated HTML is missing {entry['id']}")
    if "__" in rendered:
        raise SystemExit("generated HTML contains an unreplaced template marker")


def checked_url(value: str) -> str:
    if not value.startswith(("https://", "http://")):
        raise SystemExit(f"unsupported external URL {value!r}")
    return value


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path):
    return json.loads(path.read_text())


PAGE = """<!doctype html>
<!-- Generated by scripts/generate-plex-policy-mapping-report.py; do not edit. -->
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>PLEX v__CONTRACT__ Policy Mapping Atlas</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #65708a;
      --paper: #f7f8fc;
      --panel: rgba(255, 255, 255, 0.92);
      --line: #dce1ec;
      --shadow: 0 18px 60px rgba(28, 39, 67, 0.12);
      --admit: #7c3aed;
      --route: #2563eb;
      --schedule: #0891b2;
      --cache: #059669;
      --feedback: #d97706;
      --pass: #047857;
      --declared: #b45309;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 4%, rgba(124, 58, 237, 0.13), transparent 24rem),
        radial-gradient(circle at 92% 7%, rgba(8, 145, 178, 0.12), transparent 28rem),
        var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    a { color: inherit; }
    button, input, select { font: inherit; }
    .page { width: min(1760px, calc(100% - 40px)); margin: 0 auto; }
    .hero { padding: 64px 0 30px; }
    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      margin: 0 0 16px;
      color: #5b21b6;
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .eyebrow::before {
      width: 28px;
      height: 3px;
      content: "";
      border-radius: 99px;
      background: linear-gradient(90deg, var(--admit), var(--schedule));
    }
    h1 {
      max-width: 980px;
      margin: 0;
      font-size: clamp(2.6rem, 5.7vw, 6.6rem);
      line-height: 0.94;
      letter-spacing: -0.055em;
    }
    .hero-copy {
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(300px, 0.8fr);
      gap: 42px;
      align-items: end;
      margin-top: 32px;
    }
    .hero-copy > p {
      max-width: 850px;
      margin: 0;
      color: var(--muted);
      font-size: clamp(1.05rem, 1.7vw, 1.35rem);
    }
    .headline-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
    }
    .headline-stat {
      padding: 18px;
      border: 1px solid rgba(124, 58, 237, 0.14);
      border-radius: 18px;
      background: var(--panel);
      box-shadow: 0 8px 30px rgba(28, 39, 67, 0.06);
    }
    .headline-stat strong {
      display: block;
      font-size: 1.8rem;
      line-height: 1;
    }
    .headline-stat span {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 0.74rem;
      font-weight: 750;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .section {
      margin: 24px 0;
      padding: 28px;
      border: 1px solid rgba(89, 100, 128, 0.16);
      border-radius: 26px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .section-heading {
      display: flex;
      gap: 28px;
      align-items: end;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .section-heading h2 {
      margin: 0;
      font-size: clamp(1.5rem, 2.6vw, 2.35rem);
      letter-spacing: -0.035em;
    }
    .section-heading p {
      max-width: 720px;
      margin: 0;
      color: var(--muted);
    }
    .lifecycle {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
    }
    .coverage-card {
      position: relative;
      min-height: 164px;
      padding: 18px;
      overflow: hidden;
      color: var(--ink);
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.74);
      cursor: pointer;
      transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
    }
    .coverage-card:hover, .coverage-card[aria-pressed="true"] {
      transform: translateY(-3px);
      border-color: currentColor;
      box-shadow: 0 14px 30px rgba(28, 39, 67, 0.12);
    }
    .coverage-card[aria-pressed="true"]::after {
      position: absolute;
      top: 12px;
      right: 13px;
      content: "FILTER";
      font-size: 0.58rem;
      font-weight: 900;
      letter-spacing: 0.1em;
    }
    .coverage-name {
      display: block;
      font-size: 0.76rem;
      font-weight: 850;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .coverage-card strong {
      display: block;
      margin: 8px 0 10px;
      font-size: 2.1rem;
      line-height: 1;
    }
    .coverage-card strong small { color: var(--muted); font-size: 0.9rem; }
    .coverage-bar {
      display: block;
      height: 5px;
      margin-bottom: 12px;
      overflow: hidden;
      border-radius: 99px;
      background: #e7eaf2;
    }
    .coverage-bar i { display: block; height: 100%; background: currentColor; }
    .coverage-authority { color: var(--muted); font-size: 0.76rem; }
    .hook-admit { --hook-color: var(--admit); }
    .hook-route { --hook-color: var(--route); }
    .hook-schedule { --hook-color: var(--schedule); }
    .hook-cache { --hook-color: var(--cache); }
    .hook-feedback { --hook-color: var(--feedback); }
    .coverage-card[class*="hook-"], .filter-chip[class*="hook-"] { color: var(--hook-color); }
    .insights {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 16px;
    }
    .insight {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(247, 248, 252, 0.82);
    }
    .insight strong { display: block; margin-bottom: 5px; font-size: 1.05rem; }
    .insight p { margin: 0; color: var(--muted); font-size: 0.86rem; }
    .controls {
      position: sticky;
      top: 10px;
      z-index: 20;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(260px, auto) auto;
      gap: 12px;
      align-items: center;
      margin: 0 0 18px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(250, 251, 254, 0.94);
      box-shadow: 0 8px 30px rgba(28, 39, 67, 0.09);
      backdrop-filter: blur(18px);
    }
    .search-wrap { position: relative; }
    .search-wrap::before {
      position: absolute;
      top: 50%;
      left: 14px;
      content: "/";
      color: var(--muted);
      font-weight: 900;
      transform: translateY(-50%);
    }
    .search-wrap input, .controls select {
      width: 100%;
      min-height: 42px;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 12px;
      outline: none;
      background: white;
    }
    .search-wrap input { padding: 10px 14px 10px 34px; }
    .controls select { padding: 8px 34px 8px 12px; }
    .search-wrap input:focus, .controls select:focus { border-color: var(--route); }
    .control-actions { display: flex; gap: 8px; }
    .utility-button {
      min-height: 42px;
      padding: 8px 13px;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      cursor: pointer;
    }
    .filter-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 14px;
    }
    .filter-row > strong {
      margin-right: 4px;
      color: var(--muted);
      font-size: 0.76rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .filter-chip {
      padding: 7px 11px;
      border: 1px solid currentColor;
      border-radius: 999px;
      background: transparent;
      cursor: pointer;
      font-size: 0.78rem;
      font-weight: 750;
    }
    .filter-chip span { opacity: 0.68; }
    .filter-chip[aria-pressed="true"] { color: white; background: var(--hook-color); }
    .visible-count { margin-left: auto; color: var(--muted); font-size: 0.82rem; }
    .matrix-wrap {
      max-height: 74vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: white;
    }
    table {
      width: 100%;
      min-width: 1480px;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.82rem;
    }
    th, td {
      padding: 13px 12px;
      vertical-align: top;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }
    thead th {
      position: sticky;
      top: 0;
      z-index: 6;
      color: var(--ink);
      text-align: left;
      background: #f2f4f9;
      box-shadow: 0 1px 0 var(--line);
    }
    thead th:first-child {
      left: 0;
      z-index: 8;
      width: 330px;
      min-width: 330px;
    }
    thead th:last-child { width: 150px; min-width: 150px; }
    .hook-heading { width: 190px; min-width: 190px; border-top: 4px solid var(--hook-color); }
    .hook-heading span {
      display: block;
      color: var(--hook-color);
      font-size: 0.78rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hook-heading small {
      display: block;
      margin-top: 5px;
      color: var(--muted);
      font-size: 0.67rem;
      font-weight: 500;
      line-height: 1.35;
    }
    tbody tr.policy-row:hover > * { background-color: #fafbfe; }
    .paper-cell {
      position: sticky;
      left: 0;
      z-index: 3;
      display: flex;
      gap: 10px;
      min-width: 330px;
      text-align: left;
      background: white;
    }
    .expand-button {
      flex: 0 0 26px;
      width: 26px;
      height: 26px;
      padding: 0;
      color: var(--route);
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #f8fafc;
      cursor: pointer;
      font-weight: 900;
      line-height: 24px;
    }
    .expand-button[aria-expanded="true"] { color: white; background: var(--route); }
    .paper-copy { display: block; min-width: 0; }
    .paper-name {
      display: inline;
      color: #1d4ed8;
      font-size: 0.96rem;
      font-weight: 850;
      text-decoration: none;
    }
    .paper-name:hover { text-decoration: underline; }
    .paper-title {
      display: -webkit-box;
      margin-top: 3px;
      overflow: hidden;
      color: var(--ink);
      font-size: 0.72rem;
      font-weight: 500;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .paper-meta { display: block; margin-top: 6px; color: var(--muted); font-size: 0.68rem; }
    .paper-meta a { color: var(--route); }
    .hook-cell {
      position: relative;
      min-width: 190px;
      background: white;
    }
    .hook-cell.active {
      background: color-mix(in srgb, var(--hook-color) 7%, white);
      box-shadow: inset 3px 0 0 var(--hook-color);
    }
    .hook-cell.inactive {
      color: #b2bac9;
      text-align: center;
      background: #fbfcfe;
    }
    .hook-cell.inactive span { display: block; margin-top: 22px; }
    .mapping-kind {
      display: inline-block;
      padding: 3px 7px;
      color: var(--hook-color);
      border: 1px solid currentColor;
      border-radius: 999px;
      font-size: 0.58rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .kind-declared {
      background: #fffaf0 !important;
      box-shadow: inset 3px 0 0 var(--declared) !important;
    }
    .kind-declared .mapping-kind { color: var(--declared); }
    .hook-cell p { margin: 8px 0 0; line-height: 1.35; }
    .evidence-cell { min-width: 150px; background: white; }
    .evidence-cell strong, .evidence-cell small { display: block; margin-top: 6px; }
    .evidence-cell small { color: var(--muted); line-height: 1.25; }
    .status-pass {
      display: inline-block;
      padding: 3px 7px;
      color: white;
      border-radius: 999px;
      background: var(--pass);
      font-size: 0.6rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .detail-row td { padding: 0; background: #f6f8fc; }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(220px, 1fr));
      gap: 24px;
      padding: 24px 28px 28px 54px;
      border-left: 4px solid var(--route);
    }
    .detail-grid h3 {
      margin: 0 0 8px;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .detail-grid p { margin: 7px 0; color: var(--muted); font-size: 0.78rem; }
    .detail-grid ul { margin: 0; padding-left: 18px; }
    .detail-grid li { margin: 4px 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip {
      padding: 4px 7px;
      border-radius: 7px;
      background: #e9edf5;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.67rem;
    }
    .chip-mechanic { color: #7c2d12; background: #ffedd5; }
    .empty-value { color: var(--muted); font-style: italic; }
    .detail-links { display: flex; flex-wrap: wrap; gap: 7px; }
    .detail-links a, .muted-link {
      padding: 4px 7px;
      border: 1px solid #cbd5e1;
      border-radius: 7px;
      background: white;
      font-size: 0.68rem;
      text-decoration: none;
    }
    .muted-link { color: var(--muted); }
    .method-note {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 28px;
      align-items: center;
      margin: 20px 0 60px;
      padding: 22px 26px;
      color: #f8fafc;
      border-radius: 22px;
      background: #172033;
    }
    .method-note h2 { margin: 0 0 7px; font-size: 1.15rem; }
    .method-note p { max-width: 1000px; margin: 0; color: #cbd5e1; }
    .method-note code { color: #fde68a; }
    .method-note .mechanics-summary {
      max-width: 430px;
      color: #a5b4fc;
      font-size: 0.75rem;
      text-align: right;
    }
    [hidden] { display: none !important; }
    @media (max-width: 1050px) {
      .hero-copy { grid-template-columns: 1fr; }
      .lifecycle { grid-template-columns: repeat(2, 1fr); }
      .insights { grid-template-columns: 1fr; }
      .controls { position: static; grid-template-columns: 1fr; }
      .detail-grid { grid-template-columns: repeat(2, 1fr); }
      .method-note { grid-template-columns: 1fr; }
      .method-note .mechanics-summary { text-align: left; }
    }
    @media (max-width: 650px) {
      .page { width: min(100% - 20px, 1760px); }
      .hero { padding-top: 34px; }
      .headline-stats { grid-template-columns: 1fr; }
      .lifecycle { grid-template-columns: 1fr; }
      .section { padding: 18px; border-radius: 18px; }
      .section-heading { display: block; }
      .section-heading p { margin-top: 8px; }
      .detail-grid { grid-template-columns: 1fr; }
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --ink: #edf1fb;
        --muted: #aab3c8;
        --paper: #0e1422;
        --panel: rgba(23, 32, 51, 0.9);
        --line: #303b53;
      }
      .coverage-card, .insight, .controls, .search-wrap input, .controls select,
      .utility-button, .matrix-wrap, .paper-cell, .hook-cell, .evidence-cell,
      .detail-links a, .muted-link { background-color: #172033; color: var(--ink); }
      thead th { background: #202a40; }
      .hook-cell.active { background: color-mix(in srgb, var(--hook-color) 13%, #172033); }
      .hook-cell.inactive { background: #131b2c; }
      tbody tr.policy-row:hover > * { background-color: #202a40; }
      .detail-row td, .detail-grid { background: #111a2b; }
      .coverage-bar, .chip { background: #303b53; }
      .kind-declared { background: #2b2114 !important; }
    }
    @media print {
      :root { --paper: white; --panel: white; --ink: #111827; --muted: #4b5563; }
      body { background: white; }
      .page { width: 100%; }
      .hero { padding: 18px 0; }
      h1 { font-size: 34pt; }
      .controls, .filter-row, .expand-button, .detail-row, .method-note { display: none !important; }
      .section { padding: 12px; border: 0; box-shadow: none; }
      .matrix-wrap { max-height: none; overflow: visible; border: 0; }
      table { min-width: 100%; font-size: 7pt; }
      thead th, .paper-cell { position: static; }
      th, td { padding: 5px; }
      .hook-heading, .hook-cell { min-width: 0; width: auto; }
      .paper-cell { min-width: 0; }
      .hook-cell p { margin-top: 4px; }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <p class="eyebrow">PLEX v__CONTRACT__ presentation report</p>
      <h1>__TOTAL__ policy kernels.<br>One five-hook waist.</h1>
      <div class="hero-copy">
        <p>This atlas shows how every committed PLEX paper replication maps onto
          <strong>admit</strong>, <strong>route</strong>, <strong>schedule</strong>,
          <strong>cache</strong>, and <strong>feedback</strong>. Each filled cell
          names the policy authority exercised by the Wasm component; links lead
          to the paper, implementation, metadata, and deterministic evidence.</p>
        <div class="headline-stats">
          <div class="headline-stat"><strong>__TOTAL__</strong><span>paper policies</span></div>
          <div class="headline-stat"><strong>__PASSING__/__TOTAL__</strong><span>passing fixtures</span></div>
          <div class="headline-stat"><strong>__MULTI_HOOK__</strong><span>multi-hook policies</span></div>
          <div class="headline-stat"><strong>__DECLARED_ONLY__</strong><span>declared-only cells</span></div>
        </div>
      </div>
    </header>

    <section class="section">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Programming model coverage</p>
          <h2>The stable waist, measured by real policy kernels</h2>
        </div>
        <p>The normal path is <strong>admit &rarr; route &rarr; schedule &rarr;
          feedback</strong>. Cache is an independent decision boundary for
          insertion, pressure, expiry, and dependency progress. Click a card to
          filter the matrix.</p>
      </div>
      <div class="lifecycle">__COVERAGE_CARDS__</div>
      <div class="insights">
        <article class="insight">
          <strong>Schedule is the broadest authority.</strong>
          <p>__SCHEDULE_COUNT__ policies allocate service, often composed with feedback for
            fairness or cache for locality.</p>
        </article>
        <article class="insight">
          <strong>Admission stays deliberately narrow.</strong>
          <p>Only __ADMIT_COUNT__ kernels need request-entry authority; overload, SLO, quota,
            and branch-control policies justify the hook.</p>
        </article>
        <article class="insight">
          <strong>Feedback depth is visible, not hidden.</strong>
          <p>__CUSTOM_FEEDBACK__ policies contain custom feedback reducers;
            __DECLARED_ONLY__ feedback surfaces are declared in metadata but
            remain explicit implementation gaps.</p>
        </article>
      </div>
    </section>

    <section class="section" id="matrix">
      <div class="section-heading">
        <div>
          <p class="eyebrow">Policy-to-hook matrix</p>
          <h2>Every implemented paper, mapped cell by cell</h2>
        </div>
        <p>A <strong>decision</strong> cell returns a plan. A
          <strong>state update</strong> cell reduces enacted outcomes. A
          <strong>declared only</strong> cell appears in replication metadata
          but has no custom hook reducer in the current component; it is shown
          as a gap rather than completed semantics.</p>
      </div>
      <div class="controls">
        <label class="search-wrap">
          <input id="policy-search" type="search"
                 placeholder="Search paper, kernel, fact, mechanic, venue...">
        </label>
        <select id="category-filter" aria-label="Filter by paper category">
          <option value="">All paper categories</option>
          __CATEGORY_OPTIONS__
        </select>
        <div class="control-actions">
          <button class="utility-button" id="reset-filters" type="button">Reset</button>
          <button class="utility-button" id="print-report" type="button">Print / PDF</button>
        </div>
      </div>
      <div class="filter-row">
        <strong>Must include</strong>
        __FILTER_BUTTONS__
        <span class="visible-count"><strong id="visible-count">__TOTAL__</strong> / __TOTAL__ policies</span>
      </div>
      <div class="matrix-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Paper policy</th>
              __OPERATION_HEADERS__
              <th scope="col">Evidence</th>
            </tr>
          </thead>
          <tbody id="policy-body">__ROWS__</tbody>
        </table>
      </div>
    </section>

    <aside class="method-note">
      <div>
        <h2>Read the claim conservatively.</h2>
        <p>All __TOTAL__ entries are <code>policy-kernel-reproduction</code>
          artifacts with deterministic passing fixtures. They demonstrate that
          the PLEX v__CONTRACT__ programming model can express each central
          decision kernel. They do not claim complete source-system performance
          or physical engine mechanics.</p>
      </div>
      <div class="mechanics-summary">
        Optional mechanics represented in this corpus:
        <strong>__OPTIONAL_MECHANICS__</strong>
      </div>
    </aside>
  </main>
  <script>
    (function () {
      "use strict";
      var rows = Array.prototype.slice.call(document.querySelectorAll(".policy-row"));
      var selectedHooks = new Set();
      var search = document.getElementById("policy-search");
      var category = document.getElementById("category-filter");
      var count = document.getElementById("visible-count");
      var hookButtons = Array.prototype.slice.call(
        document.querySelectorAll("[data-hook-filter]")
      );

      function closeDetail(row) {
        var detail = row.nextElementSibling;
        var button = row.querySelector(".expand-button");
        detail.hidden = true;
        button.setAttribute("aria-expanded", "false");
        button.textContent = "+";
      }

      function applyFilters() {
        var query = search.value.trim().toLowerCase();
        var selectedCategory = category.value;
        var visible = 0;
        rows.forEach(function (row) {
          var operations = row.dataset.ops.split(" ");
          var hasHooks = Array.from(selectedHooks).every(function (hook) {
            return operations.indexOf(hook) !== -1;
          });
          var matchesSearch = !query || row.dataset.search.indexOf(query) !== -1;
          var matchesCategory =
            !selectedCategory || row.dataset.category === selectedCategory;
          var show = hasHooks && matchesSearch && matchesCategory;
          row.hidden = !show;
          closeDetail(row);
          if (show) {
            visible += 1;
          }
        });
        count.textContent = String(visible);
      }

      function syncHookButtons() {
        hookButtons.forEach(function (button) {
          button.setAttribute(
            "aria-pressed",
            selectedHooks.has(button.dataset.hookFilter) ? "true" : "false"
          );
        });
      }

      hookButtons.forEach(function (button) {
        button.addEventListener("click", function () {
          var hook = button.dataset.hookFilter;
          if (selectedHooks.has(hook)) {
            selectedHooks.delete(hook);
          } else {
            selectedHooks.add(hook);
          }
          syncHookButtons();
          applyFilters();
          document.getElementById("matrix").scrollIntoView({block: "start"});
        });
      });

      rows.forEach(function (row) {
        var button = row.querySelector(".expand-button");
        button.addEventListener("click", function () {
          var detail = row.nextElementSibling;
          var open = button.getAttribute("aria-expanded") === "true";
          detail.hidden = open;
          button.setAttribute("aria-expanded", open ? "false" : "true");
          button.textContent = open ? "+" : "-";
        });
      });

      search.addEventListener("input", applyFilters);
      category.addEventListener("change", applyFilters);
      document.getElementById("reset-filters").addEventListener("click", function () {
        selectedHooks.clear();
        search.value = "";
        category.value = "";
        syncHookButtons();
        applyFilters();
      });
      document.getElementById("print-report").addEventListener("click", function () {
        window.print();
      });
      applyFilters();
    }());
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
