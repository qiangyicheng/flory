"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import shutil
import subprocess as sp
import sys
from pathlib import Path

import pytest

PACKAGE_PATH = Path(__file__).resolve().parents[1]
EXAMPLES = (PACKAGE_PATH / "examples").glob("*.py")

@pytest.mark.slow
@pytest.mark.no_cover
@pytest.mark.parametrize("path", EXAMPLES)
def test_example_scripts(path, tmp_path):
    """Runs an example script given by path."""
    # check whether this test needs to be run
    if path.name.startswith("_"):
        pytest.skip("skip examples starting with an underscore")

    # copy the example to a temporary path
    script = tmp_path / path.name
    shutil.copy(path, script)

    # run the actual test in a separate python process
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_PATH) + ":" + env.get("PYTHONPATH", "")
    proc = sp.Popen([sys.executable, script], env=env, stdout=sp.PIPE, stderr=sp.PIPE)
    try:
        outs, errs = proc.communicate(timeout=30)
    except sp.TimeoutExpired:
        proc.kill()
        outs, errs = proc.communicate()

    # prepare output
    msg = f"Script `{path}` failed with following output:"
    if outs:
        msg = f"{msg}\nSTDOUT:\n{outs}"
    if errs:
        msg = f"{msg}\nSTDERR:\n{errs}"
    assert proc.returncode <= 0, msg
