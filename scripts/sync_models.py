"""Regenerate keycycle's model YAMLs from each provider's own published source.

    python scripts/sync_models.py                        # groq, cerebras, openrouter
    python scripts/sync_models.py --gemini-rows rows.json  # + gemini limits (from scripts/gemini_limits.sh)

Sources (all public, no login):
  groq        https://console.groq.com/docs/rate-limits.md
  cerebras    https://inference-docs.cerebras.ai/support/rate-limits.md  (Free Trial tab)
  openrouter  https://openrouter.ai/api/v1/models                        (":free" models)
  gemini      AI Studio rate-limit page (login, so scraped locally by autobrowse);
              with GEMINI_API_KEY set, models the API no longer lists are pruned,
              so deprecations land even on days without a fresh scrape.

Hour/day fields a provider does not publish are derived the way keycycle always
has: per hour = per minute * 60 (capped by per day), per day = per minute * 1440.
Files are only rewritten when content changes, and carry no date, so an
unchanged day is an empty diff.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import yaml

MODELS = Path(__file__).resolve().parents[1] / "keycycle" / "keycycle" / "config" / "models"
FIELDS = ("requests_per_minute", "requests_per_hour", "requests_per_day",
          "tokens_per_minute", "tokens_per_hour", "tokens_per_day")


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "keycycle-sync", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def num(s):
    """'1.2K' -> 1200, '1M' -> 1000000, '-' / '' / 'Unlimited' -> None."""
    s = s.strip().strip("`").replace("\\", "").replace(",", "")
    m = re.fullmatch(r"([\d.]+)\s*([KMB]?)", s, re.I)
    if not m:
        return None
    return int(round(float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()]))


def limits(rpm, rpd=None, tpm=None, tph=None, tpd=None):
    rpd = rpd or rpm * 1440
    out = {"requests_per_minute": rpm, "requests_per_hour": min(rpm * 60, rpd), "requests_per_day": rpd}
    if tpm:
        out["tokens_per_minute"] = tpm
        out["tokens_per_hour"] = tph or (min(tpm * 60, tpd) if tpd else tpm * 60)
    if tpd:
        out["tokens_per_day"] = tpd
    return out


def md_rows(md):
    """Markdown table rows as cell lists, header/separator rows dropped."""
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("|") and not re.fullmatch(r"[|\s:-]+", line):
            yield [c.strip() for c in line.strip("|").split("|")]


def groq():
    out = {}
    for c in md_rows(fetch("https://console.groq.com/docs/rate-limits.md")):
        if len(c) == 7 and num(c[1]):  # MODEL ID | RPM | RPD | TPM | TPD | ASH | ASD
            out[c[0]] = limits(num(c[1]), num(c[2]), num(c[3]), tpd=num(c[4]))
    return out


def cerebras():
    md = fetch("https://inference-docs.cerebras.ai/support/rate-limits.md")
    free = md.split('<Tab title="Free Trial">', 1)[1].split("</Tab>", 1)[0]
    out = {}
    for c in md_rows(free):
        if len(c) == 6 and num(c[1]):  # Model | RPM | Uncached TPM | Total TPM | TPH | TPD
            out[re.sub(r"<sup>.*?</sup>|`", "", c[0]).strip()] = limits(
                num(c[1]), tpm=num(c[2]), tph=num(c[4]), tpd=num(c[5]))
    if out:  # unlisted models: the tightest published limits
        out["default"] = {f: min(m[f] for m in out.values()) for f in next(iter(out.values()))}
    return out


def openrouter():
    data = json.loads(fetch("https://openrouter.ai/api/v1/models"))["data"]
    return [{"name": re.sub(r"\s*\(free\)$", "", m["name"].split(": ", 1)[-1]).strip(),
             "id": m["id"], "max_context_length": m.get("context_length")}
            for m in data if m["id"].endswith(":free")]


NOISE = re.compile(r"^(preview|it|latest|a\d+b|\d{2}|\d{4})$")


def norm(name):
    """'Gemini 3 Flash' and 'gemini-3-flash-preview' -> 'gemini-3-flash'; dates, -it, -a4b dropped."""
    toks = re.split(r"[\s_-]+", re.sub(r"\(.*?\)", "", name).lower())
    return "-".join(t for t in toks if t and not NOISE.match(t))


def gemini_api_models(key):
    names, tok = [], ""
    while True:
        r = json.loads(fetch(f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&pageToken={tok}",
                             {"x-goog-api-key": key}))
        names += [m["name"].split("/", 1)[1] for m in r.get("models", [])]
        tok = r.get("nextPageToken")
        if not tok:
            return names


def gemini(rows_file, key):
    current = yaml.safe_load((MODELS / "gemini.yaml").read_text()) or {}
    api = gemini_api_models(key) if key else None
    if rows_file:
        if api is None:
            sys.exit("--gemini-rows needs GEMINI_API_KEY to map AI Studio names to model ids")
        out, text = {}, []
        for c in json.loads(Path(rows_file).read_text()):
            # [status, Model, Category, 'used / RPM', 'used / TPM', 'used / RPD', charts]
            if len(c) < 6 or "grounding" in c[2].lower() or "live" in c[2].lower():
                continue
            rpm, tpm, rpd = (num(x.split("/")[-1]) if "/" in x else None for x in c[3:6])
            if not rpm or not rpd:  # 0 / 0 = not on the free tier
                continue
            for mid in api:
                if norm(mid) == norm(c[1]):
                    out[mid] = limits(rpm, rpd, tpm)
                    if c[2] == "Text-out models":
                        text.append(out[mid])
        if out:  # unlisted models and the -latest aliases: the tightest text-out limits
            if text:
                out["default"] = {f: min(v[f] for v in text) for f in text[0]}
            current = out
    elif api is not None:  # no fresh scrape: still drop what Google retired
        current = {k: v for k, v in current.items() if k == "default" or k in api}
    return current


def write(name, data, source):
    path = MODELS / name
    body = f"# Generated by scripts/sync_models.py from {source}. Hand edits are overwritten.\n\n"
    body += yaml.safe_dump(data, sort_keys=False, width=200)
    if not data:
        print(f"{name}: source returned nothing, left as is", file=sys.stderr)
    elif path.read_text() != body:
        path.write_text(body)
        print(f"{name}: updated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini-rows", help="JSON rows from scripts/gemini_limits.sh")
    a = ap.parse_args()
    failed = False
    for name, fn, src in (
        ("groq.yaml", groq, "https://console.groq.com/docs/rate-limits.md"),
        ("cerebras.yaml", cerebras, "https://inference-docs.cerebras.ai/support/rate-limits.md (Free Trial)"),
        ("openrouter_models.yaml", openrouter, "https://openrouter.ai/api/v1/models (:free)"),
        ("gemini.yaml", lambda: gemini(a.gemini_rows, os.getenv("GEMINI_API_KEY")),
         "https://aistudio.google.com/rate-limit (free tier) + the Gemini models API"),
    ):
        try:
            write(name, fn(), src)
        except Exception as e:  # one provider's page changing shape must not block the rest
            failed = True
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
