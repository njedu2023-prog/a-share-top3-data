from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


IGNORE_PARTS = {"__pycache__", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}
MAX_REQUEST_ATTEMPTS = 6
TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}


class GitHubRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


def retry_delay(attempt: int, retry_after: str = "") -> int:
    if retry_after.isdigit():
        return min(max(int(retry_after), 1), 60)
    return min(2 * (2**attempt), 30)


def concise_error_body(body: str, limit: int = 1000) -> str:
    compact = " ".join(body.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


def request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    for attempt in range(MAX_REQUEST_ATTEMPTS):
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
            rate_limited = exc.code == 403 and (
                "secondary rate limit" in body.lower()
                or exc.headers.get("X-RateLimit-Remaining", "") == "0"
            )
            retryable = exc.code in TRANSIENT_HTTP_CODES or rate_limited
            if retryable and attempt < MAX_REQUEST_ATTEMPTS - 1:
                delay = retry_delay(attempt, exc.headers.get("Retry-After", ""))
                print(
                    f"GitHub API HTTP {exc.code}; retry "
                    f"{attempt + 2}/{MAX_REQUEST_ATTEMPTS} in {delay}s.",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise GitHubRequestError(
                f"{method} {url} failed: {exc.code} {concise_error_body(body)}",
                retryable=retryable,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt < MAX_REQUEST_ATTEMPTS - 1:
                delay = retry_delay(attempt)
                print(
                    f"GitHub API network error; retry "
                    f"{attempt + 2}/{MAX_REQUEST_ATTEMPTS} in {delay}s: {exc}",
                    flush=True,
                )
                time.sleep(delay)
                continue
            raise GitHubRequestError(
                f"{method} {url} failed after network retries: {exc}",
                retryable=True,
            ) from exc
    raise GitHubRequestError(f"{method} {url} failed after retries", retryable=True)


def iter_files(paths: list[str]):
    cwd = Path.cwd().resolve()
    seen: set[str] = set()
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
            rel_posix = rel.as_posix()
            if rel_posix in seen:
                continue
            seen.add(rel_posix)
            yield path, rel_posix


def is_binary(path: Path) -> bool:
    guess, _ = mimetypes.guess_type(path.name)
    if guess and not guess.startswith("text/"):
        return True
    try:
        path.read_text(encoding="utf-8")
        return False
    except UnicodeDecodeError:
        return True


def commit_via_git_data_api(
    files: list[tuple[Path, str]], repo: str, branch: str, message: str, token: str
) -> str | None:
    ref = request("GET", f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}", token)
    base_commit_sha = ref["object"]["sha"]
    base_commit = request("GET", f"https://api.github.com/repos/{repo}/git/commits/{base_commit_sha}", token)
    base_tree_sha = base_commit["tree"]["sha"]

    tree = []
    for path, rel in files:
        if is_binary(path):
            content = base64.b64encode(path.read_bytes()).decode("ascii")
            blob = request(
                "POST",
                f"https://api.github.com/repos/{repo}/git/blobs",
                token,
                {"content": content, "encoding": "base64"},
            )
            tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        else:
            content = path.read_text(encoding="utf-8")
            tree.append({"path": rel, "mode": "100644", "type": "blob", "content": content})

    new_tree = request(
        "POST",
        f"https://api.github.com/repos/{repo}/git/trees",
        token,
        {"base_tree": base_tree_sha, "tree": tree},
    )
    if new_tree["sha"] == base_tree_sha:
        print("No changed files to commit.")
        return None

    commit = request(
        "POST",
        f"https://api.github.com/repos/{repo}/git/commits",
        token,
        {"message": message, "tree": new_tree["sha"], "parents": [base_commit_sha]},
    )
    request("PATCH", f"https://api.github.com/repos/{repo}/git/refs/heads/{branch}", token, {"sha": commit["sha"]})
    return str(commit["sha"])


def graphql_request(query: str, variables: dict, token: str) -> dict:
    result = request(
        "POST",
        "https://api.github.com/graphql",
        token,
        {"query": query, "variables": variables},
    )
    errors = result.get("errors") or []
    if errors:
        error_text = concise_error_body(json.dumps(errors, ensure_ascii=False))
        raise GitHubRequestError(f"GitHub GraphQL failed: {error_text}", retryable=False)
    return result["data"]


def commit_via_graphql(
    files: list[tuple[Path, str]], repo: str, branch: str, message: str, token: str
) -> str:
    owner, name = repo.split("/", 1)
    head_data = graphql_request(
        """
        query($owner: String!, $name: String!, $ref: String!) {
          repository(owner: $owner, name: $name) {
            ref(qualifiedName: $ref) { target { oid } }
          }
        }
        """,
        {"owner": owner, "name": name, "ref": f"refs/heads/{branch}"},
        token,
    )
    expected_head = head_data["repository"]["ref"]["target"]["oid"]
    additions = [
        {
            "path": rel,
            "contents": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        for path, rel in files
    ]
    commit_data = graphql_request(
        """
        mutation($input: CreateCommitOnBranchInput!) {
          createCommitOnBranch(input: $input) { commit { oid } }
        }
        """,
        {
            "input": {
                "branch": {"repositoryNameWithOwner": repo, "branchName": branch},
                "message": {"headline": message},
                "expectedHeadOid": expected_head,
                "fileChanges": {"additions": additions},
            }
        },
        token,
    )
    return str(commit_data["createCommitOnBranch"]["commit"]["oid"])


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: github_commit_paths.py commit_message path [path ...]")
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ.get("GITHUB_REF_NAME", "main")
    message = sys.argv[1]
    paths = sys.argv[2:]

    files = list(iter_files(paths))
    if not files:
        print("No files found for API commit.")
        return
    try:
        commit_sha = commit_via_git_data_api(files, repo, branch, message, token)
        if commit_sha is not None:
            print(f"Committed {len(files)} files through GitHub Git Data API: {commit_sha}")
    except GitHubRequestError as exc:
        if not exc.retryable:
            raise
        print(f"Git Data API remained unavailable; switching to GraphQL: {exc}", flush=True)
        commit_sha = commit_via_graphql(files, repo, branch, message, token)
        print(f"Committed {len(files)} files atomically through GitHub GraphQL: {commit_sha}")


if __name__ == "__main__":
    main()
