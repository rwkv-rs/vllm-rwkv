# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest


@pytest.fixture
def workflow_text() -> str:
    repo_root = Path(__file__).parents[1]
    return (repo_root / ".github/workflows/rebase-rwkv.yml").read_text()


def _git_push_commands(workflow_text: str) -> list[str]:
    commands = []
    continued = ""
    for line in workflow_text.splitlines():
        stripped = line.strip()
        if continued:
            continued += " " + stripped.removesuffix("\\").rstrip()
            if not stripped.endswith("\\"):
                commands.append(continued)
                continued = ""
        elif stripped.startswith("git push"):
            continued = stripped.removesuffix("\\").rstrip()
            if not stripped.endswith("\\"):
                commands.append(continued)
                continued = ""
    assert not continued
    return commands


def test_upstream_sync_only_pushes_the_review_candidate(workflow_text: str):
    commands = _git_push_commands(workflow_text)

    assert commands
    assert 'candidate_branch="automation/rebase-rwkv"' in workflow_text
    assert all('HEAD:"$candidate_branch"' in command for command in commands)
    assert "origin HEAD:rwkv" not in workflow_text
    assert "refs/heads/rwkv:" not in workflow_text
    assert "--base rwkv" in workflow_text


def test_upstream_sync_reports_exact_conflict_evidence(workflow_text: str):
    for evidence in (
        "old_head=$old_head",
        "upstream_head=$upstream_head",
        "conflict_commit=$conflict_commit",
        "unresolved_paths<<RWKV_PATHS",
        "git diff --name-only --diff-filter=U",
        "git rebase --abort",
    ):
        assert evidence in workflow_text


def test_upstream_sync_fails_closed_for_sensitive_paths(workflow_text: str):
    assert "sensitive_paths<<RWKV_PATHS" in workflow_text
    assert "sensitive-semantics-review-required" in workflow_text
    assert "Fail closed on sensitive paths" in workflow_text
    assert "explicit human review is required" in workflow_text
