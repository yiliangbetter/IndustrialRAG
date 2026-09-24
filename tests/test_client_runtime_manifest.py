"""Contract checks for the portable client's declarative runtime bundle."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MANIFEST_PATH = ROOT / "config" / "client_runtime_manifest.json"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _local_script_dependencies(script: Path) -> set[str]:
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.partition(".")[0])
    return {
        f"{module}.py" for module in modules if (SCRIPTS / f"{module}.py").is_file()
    }


def test_client_runtime_manifest_assets_exist() -> None:
    manifest = _manifest()

    assert manifest["version"] == 1
    assert {"domain_schema.json", "env.example", "query_steering_profiles.json"} <= set(
        manifest["configs"]
    )
    for group in ("script_entrypoints", "scripts", "configs", "models"):
        entries = manifest[group]
        assert entries
        assert len(entries) == len(set(entries)), f"duplicate {group} entry"
        assert all(Path(entry).name == entry for entry in entries)
    for name in manifest["scripts"]:
        assert (SCRIPTS / name).is_file(), f"missing packaged script: {name}"
    for name in manifest["configs"]:
        assert (ROOT / "config" / name).is_file(), f"missing packaged config: {name}"


def test_client_runtime_manifest_contains_exact_import_closure() -> None:
    manifest = _manifest()
    declared = set(manifest["scripts"])
    pending = list(manifest["script_entrypoints"])
    closure: set[str] = set()

    while pending:
        name = pending.pop()
        assert name in declared, f"entry point is not packaged: {name}"
        if name in closure:
            continue
        closure.add(name)
        pending.extend(_local_script_dependencies(SCRIPTS / name) - closure)

    assert closure == declared


def test_client_packer_consumes_manifest_without_missing_readme() -> None:
    packer = (SCRIPTS / "pack_client.ps1").read_text(encoding="utf-8")

    assert "client_runtime_manifest.json" in packer
    assert "ConvertFrom-Json" in packer
    assert "docs/client/README.txt" not in packer
