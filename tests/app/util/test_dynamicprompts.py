from __future__ import annotations

from pathlib import Path

import pytest
from dynamicprompts.wildcards import WildcardManager

from invokeai.app.util.dynamicprompts import find_missing_wildcards


@pytest.fixture
def wildcard_manager(tmp_path: Path) -> WildcardManager:
    """A manager over a directory holding a single `poses.txt` wildcard."""
    (tmp_path / "poses.txt").write_text("standing\nkneeling\n", encoding="utf-8")
    return WildcardManager(tmp_path)


def test_find_missing_wildcards_detects_unknown_wildcard_in_variant() -> None:
    # Regression: `__random__` inside a variant is parsed as a wildcard reference. Left unchecked it
    # sends the combinatorial generator into an infinite loop, so it must be reported up front.
    assert find_missing_wildcards("{__random__8chan|fenster|stuff}") == ["random"]


def test_find_missing_wildcards_detects_unknown_wildcard_nested_in_sequence_in_variant() -> None:
    # The wildcard hangs the generator even when wrapped in other text inside the variant value.
    assert find_missing_wildcards("{a __nope__|b}") == ["nope"]


@pytest.mark.parametrize("prompt", ["a __nope__ b", "__nope__", "a photo, __my_style__"])
def test_find_missing_wildcards_ignores_wildcards_outside_variants(prompt: str) -> None:
    # A wildcard used as plain literal text generates fine (no hang), so it must not be reported.
    assert find_missing_wildcards(prompt) == []


@pytest.mark.parametrize("prompt", ["plain text", "{a|b|c}", "a {2$$x|y|z}"])
def test_find_missing_wildcards_ignores_prompts_without_wildcards(prompt: str) -> None:
    assert find_missing_wildcards(prompt) == []


def test_find_missing_wildcards_dedupes_repeated_unknown_wildcards() -> None:
    assert find_missing_wildcards("{__nope__|a} {__nope__|b} {__other__|c}") == ["nope", "other"]


def test_find_missing_wildcards_resolves_against_a_wildcard_manager(wildcard_manager: WildcardManager) -> None:
    # A wildcard backed by a file on disk generates fine, so it must not be reported as missing.
    assert find_missing_wildcards("{__poses__|a}", wildcard_manager) == []


def test_find_missing_wildcards_still_reports_unknown_names_with_a_manager(
    wildcard_manager: WildcardManager,
) -> None:
    # Only the names the manager cannot resolve hang the generator.
    assert find_missing_wildcards("{__poses__|a} {__nope__|b}", wildcard_manager) == ["nope"]


def test_find_missing_wildcards_without_a_manager_treats_every_wildcard_as_missing(tmp_path: Path) -> None:
    # Default behaviour resolves nothing against disk, so even a real file is reported. Callers that
    # have a wildcards directory must pass its manager in.
    (tmp_path / "poses.txt").write_text("standing\n", encoding="utf-8")
    assert find_missing_wildcards("{__poses__|a}") == ["poses"]
