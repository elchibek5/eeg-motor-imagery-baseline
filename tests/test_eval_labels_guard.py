import os

import pytest

from run_pipeline import check_eval_labels


def test_symlinked_eval_folder_is_refused(tmp_path):
    ev = tmp_path / "Evaluation"
    ev.mkdir()
    link = tmp_path / "labels"
    os.symlink(ev, link)
    with pytest.raises(ValueError, match="separate folder"):
        check_eval_labels(str(ev), str(link))


def test_eval_folder_under_another_case_is_refused(tmp_path):
    ev = tmp_path / "Evaluation"
    ev.mkdir()
    alias = tmp_path / "evaluation"
    if not alias.exists():
        pytest.skip("case-sensitive file system: the alias is a different folder")
    with pytest.raises(ValueError, match="separate folder"):
        check_eval_labels(str(ev), str(alias))
