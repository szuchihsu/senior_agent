"""
tools/diff_tools.py

Utilities for cleaning and preprocessing git diffs before embedding.

Raw git diffs contain a lot of noise that hurts embedding quality:
  - File permission metadata lines
  - Binary file markers
  - Auto-generated files (package-lock.json, *.pb.go, etc.)
  - Extremely long diffs that exceed embedding model limits

We clean the diff down to just the meaningful code changes before
passing it to the embedding model.
"""

import re


# Files we don't want to include in the diff — changes here don't
# usually cause test failures in the application code.
IGNORED_FILE_PATTERNS = [
    r"package-lock\.json",
    r"yarn\.lock",
    r"poetry\.lock",
    r"Pipfile\.lock",
    r"\.pb\.go$",          # protobuf generated files
    r"\.generated\.",      # any generated file
    r"dist/",              # build output
    r"__pycache__/",
    r"\.pyc$",
    r"\.min\.js$",         # minified JS
    r"CHANGELOG",
    r"\.svg$",             # SVG assets
]


def clean_diff(raw_diff: str, max_chars: int = 8000) -> str:
    """
    Cleans a raw git diff for use as embedding input.

    Steps:
      1. Split the diff into per-file sections
      2. Drop sections for files we don't care about (lock files, generated code)
      3. Strip git metadata lines (index, mode, similarity) — not semantically meaningful
      4. Truncate to max_chars so we don't exceed embedding model limits

    Args:
        raw_diff:  The raw string output of `git show <sha>`
        max_chars: Maximum character length of the cleaned diff.
                   Most embedding models handle ~8K chars well.

    Returns:
        A cleaner diff string focused on meaningful code changes.
        Returns empty string if nothing meaningful remains after cleaning.

    Example:
        Input:  "diff --git a/package-lock.json b/package-lock.json\\n..."
        Output: "" (empty — lock file change is ignored)
    """
    if not raw_diff:
        return ""

    # Split on "diff --git" to get per-file sections
    # re.split with a capture group keeps the delimiter in the output
    file_sections = re.split(r"(?=diff --git )", raw_diff)

    cleaned_sections = []

    for section in file_sections:
        if not section.strip():
            continue

        # Extract the filename from the diff header
        # "diff --git a/src/auth.py b/src/auth.py" → "src/auth.py"
        filename_match = re.search(r"diff --git a/(.+?) b/", section)
        if not filename_match:
            continue

        filename = filename_match.group(1)

        # Skip ignored file patterns
        if _should_ignore_file(filename):
            continue

        # Strip git metadata lines that aren't useful for semantic understanding
        # These lines start with "index ", "new file mode", "similarity index", etc.
        cleaned_section = _strip_metadata_lines(section)

        if cleaned_section.strip():
            cleaned_sections.append(cleaned_section)

    result = "\n".join(cleaned_sections)

    # Truncate if too long — take the first max_chars characters
    # This keeps the most important (usually top-of-file) changes
    if len(result) > max_chars:
        result = result[:max_chars] + "\n... (truncated)"

    return result


def extract_changed_files(raw_diff: str) -> list[str]:
    """
    Extracts just the list of filenames changed in a diff.

    Useful for quick filtering — if a diff only changes docs or tests,
    we might weight its predictions differently.

    Args:
        raw_diff: Raw git diff string

    Returns:
        List of changed file paths, e.g. ["src/auth.py", "tests/test_auth.py"]
    """
    pattern = r"diff --git a/(.+?) b/"
    return re.findall(pattern, raw_diff)


def _should_ignore_file(filename: str) -> bool:
    """Returns True if this file should be excluded from the cleaned diff."""
    for pattern in IGNORED_FILE_PATTERNS:
        if re.search(pattern, filename):
            return True
    return False


def _strip_metadata_lines(section: str) -> str:
    """
    Removes git metadata lines from a diff section.

    Lines like:
      "index a3f9c21..b2e4d89 100644"
      "new file mode 100644"
      "similarity index 95%"
      "rename from old_name.py"

    These are git internals — not meaningful for semantic similarity.
    We keep the "--- a/file", "+++ b/file", "@@" and actual +/- lines.
    """
    lines = section.split("\n")
    kept = []

    for line in lines:
        # Skip git metadata prefixes
        if re.match(r"^(index |new file mode|deleted file mode|similarity index|rename from|rename to|Binary files)", line):
            continue
        kept.append(line)

    return "\n".join(kept)
