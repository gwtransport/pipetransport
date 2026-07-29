"""Execute the docstring examples of every pipetransport module.

The Examples blocks are shipped as runnable "Try it live" snippets in the documentation, so
they are executable code that has to keep working, not prose. Every module of the package --
public and private -- is walked, its doctests are run, and a mismatch is reported with the
verbatim doctest diff (expected versus got).
"""

import doctest
import importlib
import io
import pkgutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import pipetransport

mpl.use("Agg")

# ELLIPSIS matches pytest's own default for --doctest-modules, so a docstring that passes here
# also passes when the collector picks it up. No other flag is enabled: whitespace and repr
# formatting must match exactly.
OPTIONFLAGS = doctest.ELLIPSIS

PACKAGE_DIR = Path(pipetransport.__file__).parent
MODULE_NAMES = sorted([
    pipetransport.__name__,
    *(info.name for info in pkgutil.walk_packages(pipetransport.__path__, prefix=f"{pipetransport.__name__}.")),
])


def _examples_in(module):
    """Return the number of doctest examples the finder sees in a module."""
    return sum(len(test.examples) for test in doctest.DocTestFinder().find(module, module.__name__))


def test_every_module_is_covered():
    """Every .py file of the package must be reached, private modules included."""
    on_disk = {path.resolve() for path in PACKAGE_DIR.glob("*.py")}
    walked = {Path(importlib.import_module(name).__file__).resolve() for name in MODULE_NAMES}
    assert walked == on_disk, f"discovery missed {sorted(on_disk - walked)}, invented {sorted(walked - on_disk)}"
    assert {"_transfer.py", "_validation.py"} <= {path.name for path in walked}


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_finder_sees_the_examples(module_name):
    """A module whose source carries a doctest prompt must yield doctests, and vice versa.

    Without this the runner below could pass by finding nothing at all -- for instance if a
    docstring example were indented into a place ``DocTestFinder`` does not look.
    """
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert (_examples_in(module) > 0) == (">>>" in source), (
        f"{module_name}: source prompts and collected doctests disagree"
    )


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_doctests(module_name):
    """Run every docstring example of the module and fail with the doctest report."""
    module = importlib.import_module(module_name)
    runner = doctest.DocTestRunner(optionflags=OPTIONFLAGS, verbose=False)
    report = io.StringIO()
    try:
        for test in sorted(doctest.DocTestFinder().find(module, module_name), key=lambda t: (t.name, t.lineno or 0)):
            runner.run(test, out=report.write)
    finally:
        # Docstring examples in plot.py draw into new figures; drop them so the doctests of a
        # long module cannot trip matplotlib's open-figure warning.
        plt.close("all")

    assert runner.failures == 0, (
        f"{runner.failures} of {runner.tries} docstring examples failed in {module_name}\n\n{report.getvalue()}"
    )
    # Every example the finder saw must have been executed. doctest counts an example only
    # after deciding not to skip it, so this catches a "# doctest: +SKIP" directive quietly
    # removing a published snippet from CI while the module still reports zero failures.
    assert runner.tries == _examples_in(module), f"{module_name}: ran {runner.tries} of {_examples_in(module)} examples"
