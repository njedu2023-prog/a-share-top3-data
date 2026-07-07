from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


IGNORE_PARTS = {"__pycache__", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}


def request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {body}") from exc


def iter_files(paths: list[str]):
    cwd = Path.cwd().resolve()
    for raw_path in paths:
        root = Path(raw_path).resolve()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            rel = path.resolve().relative_to(cwd)
            if any(part in IGNORE_PARTS for part in rel.parts):
                continue
            if path.suffix in IGNORE_SUFFIXES:
                continue
            yield path, rel.as_posix()


def is_binary(path: Path) -> bool:
    guess, _ = mimetypes.guess_type(path.name)
    if guess and not guess.startswith("text/"):
        return True
    try:
        path.read_text(encoding="utf-8")
        return False
    except UnicodeDecodeError:
        return True


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: github_commit_paths.py commit_message path [path ...]")
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_REF_NAME", "main")
    message = sys.argv[1]
    paths = sys.argv[2:]

    ref = request("GET", f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}", token)
    base_commit_sha = ref["object"]["sha"]
    base_commit = request("GET", f"https://api.github.com/repos/{repo}/git/commits/{base_commit_sha}", token)
    base_tree_sha = base_commit["tree"]["sha"]

    tree = []
    for path, rel in iter_files(paths):
        if is_binary(path):
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            encoding = "base64"
        else:
            content = path.read_text(encoding="utf-8")
            encoding = "utf-8"
        blob = request(
            "POST",
            f"https://api.github.com/repos/{repo}/git/blobs",
            token,
            {"content": content, "encoding": encoding},
        )
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    if not tree:
        print("No files found for API commit.")
        return

    new_tree = request(
        "POST",
        f"https://api.github.com/repos/{repo}/git/trees",
        token,
        {"base_tree": base_tree_sha, "tree": tree},
    )
    if new_tree["sha"] == base_tree_sha:
        print("No changed files to commit.")
        return

    commit = request(
        "POST",
        f"https://api.github.com/repos/{repo}/git/commits",
        token,
        {"message": message, "tree": new_tree["sha"], "parents": [base_commit_sha]},
    )
    request("PATCH", f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}", token, {"sha": commit["sha"]})
    print(f"Committed {len(tree)} files through GitHub API: {commit['sha']}")


if __name__ == "__main__":
    main()
