from pathlib import Path

from ugr16ad import paths


def test_project_root_is_repo_root():
    # parents[2] breaks silently if paths.py ever moves; pyproject.toml pins the root.
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()


def test_supplied_csvs_are_where_paths_says():
    # The CSVs are committed, so a missing file is a broken checkout, not a missing download.
    for name in ("Xtrain", "Ytrain", "Xtest", "Ytest"):
        assert (paths.UGR16_DIR / f"UGR16v2.{name}.csv").is_file()


def test_resolve_anchors_relative_paths_at_root():
    assert paths.resolve("data") == paths.DATA_DIR
    absolute = Path(paths.PROJECT_ROOT.anchor) / "elsewhere"
    assert paths.resolve(absolute) == absolute
