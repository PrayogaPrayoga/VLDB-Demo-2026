"""Headless smoke test of the demo code path in the training notebook.

Stubs the Mercury runtime (button clicked=True, widgets return their defaults),
captures display() HTML, and execs the demo-relevant cells to catch runtime
errors before relying on the live server.
"""
import sys
import types
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class _StopExecution(Exception):
    _render_traceback_ = lambda self: []


class _Widget:
    def __init__(self, value=None):
        self.value = value


class _Button:
    def __init__(self, **kw):
        self.clicked = True


class _File:
    filepath = None


captured_html = []


def _make_mr():
    mr = types.ModuleType("mercury")
    mr.StopExecution = _StopExecution
    mr.App = lambda **kw: types.SimpleNamespace(**kw)

    def Select(label=None, value=None, choices=None, **kw):
        return _Widget(value if value is not None else (choices[0] if choices else None))

    mr.Select = Select
    mr.MultiSelect = lambda label=None, value=None, choices=None, **kw: _Widget(value)
    mr.Checkbox = lambda label=None, value=True, **kw: _Widget(value)
    mr.Numeric = lambda label=None, value=0, **kw: _Widget(value)
    mr.Slider = lambda value=0, **kw: _Widget(value)
    mr.Button = lambda **kw: _Button(**kw)
    mr.File = lambda **kw: _File()
    mr.OutputDir = lambda **kw: _Widget(None)
    mr.Markdown = lambda *a, **k: None
    mr.Note = lambda *a, **k: None
    mr.Confetti = lambda *a, **k: captured_html.append("<confetti>")

    def Stop():
        raise _StopExecution()

    mr.Stop = Stop
    return mr


def _make_display():
    def display(obj):
        html = getattr(obj, "data", None) or getattr(obj, "_repr_html_", lambda: None)()
        if isinstance(obj, str):
            html = obj
        if html:
            captured_html.append(str(html))

    class HTML:
        def __init__(self, data):
            self.data = data

    return display, HTML


def run(mode):
    nb = nbformat.read(str(REPO / "certain-prep-training-view.ipynb"), as_version=4)
    mr = _make_mr()
    display, HTML = _make_display()
    ns = {"__name__": "__main__"}

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mercury":
            return mr
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        # override the URL-toggle default so we can force a mode
        import os
        os.environ["MINPREP_MODE"] = mode
        for i, cell in enumerate(nb.cells):
            if i == 1:
                continue  # skip the fixed-position explorer iframe
            src = "\n".join(
                l for l in cell.source.splitlines() if not l.strip().startswith("%")
            )
            # ensure display/HTML from our stub are in scope
            ns["display"] = display
            ns["HTML"] = HTML
            try:
                exec(compile(src, f"cell{i}", "exec"), ns)
            except _StopExecution:
                print(f"  [cell {i}] mr.Stop() reached (expected in some paths)")
                break
    finally:
        builtins.__import__ = real_import
    return ns


if __name__ == "__main__":
    for mode in ("demo",):
        captured_html.clear()
        print(f"=== MODE={mode} ===")
        ns = run(mode)
        joined = "\n".join(captured_html)
        for key in ("MinPrep Results", "Result:", "Minimal Repair", "ActiveClean",
                    "Full Imputation", "<confetti>"):
            print(f"  contains {key!r}: {key in joined}")
        print(f"  captured {len(captured_html)} html blocks")
