from pathlib import Path

def find_project_root(start_path: Path | None = None) -> Path:
    p = (Path(start_path) if start_path else Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return p