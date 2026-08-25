#!/usr/bin/env python
"""Regenerate certain-prep-training-view.ipynb.

The training view supports two runtime modes selected by a hidden URL toggle
(?mode=run) with a build-level default from MINPREP_MODE:

  * demo mode  — instant, pre-computed results from demo_results/results.json.
  * run  mode  — the original live CM/MR pipeline (kept verbatim below).

The notebook is generated from this script so the two-mode wiring stays
readable and reproducible. Run:

    python tools/build_training_notebook.py
"""
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "certain-prep-training-view.ipynb"

C_IMPORTS = r'''
import os
import sys
from pathlib import Path

APP_DIR = Path(os.environ.get('MINPREP_APP_DIR', Path.cwd())).expanduser().resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import warnings
import mercury as mr
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML

import minprep_core as mc
from minprep_core import (
    missing_values_table,
    preprocess_for_training,
    certain_clean_main,
    MR_main,
    MR_main_regression,
    SHARED_DIR,
    DEFAULT_DATASET,
    DEFAULT_MODE,
)

warnings.filterwarnings("ignore")
mr.StopExecution._render_traceback_ = lambda self: []

app = mr.App(title="MinPrep", description="Certain & Approximately Certain Models for ML data preparation", show_code=False)


def _fmt_score(value, label):
    if label == 'Accuracy':
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def metric_cards(title, accent, boxes, subtitle=None):
    sub = f"<div style='color:#6c757d;margin-top:6px;font-size:14px;'>{subtitle}</div>" if subtitle else ""
    head = (
        f"<hr style='border:none;border-top:2px solid #e9ecef;margin:24px 0;'>"
        f"<div style='padding:14px 20px;background:#f8f9fa;border-left:5px solid {accent};border-radius:0 8px 8px 0;margin-bottom:16px;'>"
        f"<h3 style='margin:0;color:#2c3e50;font-size:20px;font-weight:600;'>{title}</h3>{sub}</div>"
    )
    body = "<div style='display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 24px 0;'>"
    for value, label in boxes:
        body += (
            f"<div style='border:1px solid #e0e0e0;border-top:3px solid {accent};border-radius:8px;padding:18px 24px;text-align:center;min-width:150px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.06);'>"
            f"<div style='font-size:14px;color:#6c757d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;'>{label}</div>"
            f"<div style='font-size:32px;font-weight:700;color:#212529;'>{value}</div></div>"
        )
    body += "</div>"
    display(HTML(head + body))
'''.strip()

C_IFRAME = (
    "from IPython.display import display, HTML\n"
    "html_code = '''\n"
    "<style>\n"
    "    #iframe-container {\n"
    "        position: fixed;\n"
    "        top: 0;\n"
    "        right: 0;\n"
    "        width: 18%;\n"
    "        height: 100%;\n"
    "        z-index: 9999;\n"
    "    }\n"
    "    #iframe-container iframe {\n"
    "        width: 100%;\n"
    "        height: 100%;\n"
    "        border: none;\n"
    "    }\n"
    "</style>\n"
    '<div id="iframe-container">\n'
    '    <iframe src="http://127.0.0.1:8000/app/certain-prep-explorer-view"></iframe>\n'
    "</div>\n"
    "<script>\n"
    '    document.querySelector("#iframe-container iframe").addEventListener("click", function() {\n'
    "        window.open(this.src, '_blank');\n"
    "    });\n"
    "</script>\n"
    "'''\n\n"
    "display(HTML(html_code))"
)

C_MODE = r'''
# Hidden presenter toggle: default is demo mode; open the page with ?mode=run
# to silently switch to run mode. The widget is hidden from the audience.
mode_widget = mr.Select(label="Mode", value=DEFAULT_MODE, choices=["demo", "run"], url_key="mode", hidden=True)
MODE = (mode_widget.value or DEFAULT_MODE).lower()
DEMO = MODE == "demo"
'''.strip()

C_INPUT = r'''
if DEMO:
    _ds_choices = mc.demo_datasets()
    _ds_label_to_key = {label: key for key, label in _ds_choices}
    dataset_sel = mr.Select(label="Dataset", value=_ds_choices[0][1], choices=[label for _, label in _ds_choices])
    dataset_key = _ds_label_to_key[dataset_sel.value]
    _meta = mc.demo_dataset_meta(dataset_key)
    df = pd.read_csv(mc.demo_dataset_path(dataset_key))
    data_file = None
else:
    data_file = mr.File(label="Upload Dataset", max_file_size="10MB")
    df = pd.read_csv(DEFAULT_DATASET)
'''.strip()

C_SYNC = r'''
if DEMO:
    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SHARED_DIR / "data.csv", index=False)
else:
    if data_file.filepath is None:
        mr.Stop()
    else:
        df = pd.read_csv(data_file.filepath)
        SHARED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SHARED_DIR / "data.csv", index=False)
'''.strip()

C_CONFIG = r'''
if DEMO:
    task = _meta['task']
    target = _meta['target']

    _ = mr.Note("**MinPrep pipeline** &nbsp;·&nbsp; Step 1: check for a (Approximately) Certain Model &nbsp;→&nbsp; Step 2: if none exists, repair the data. Pick the approach below and press **Run**.")

    _model_choices = mc.demo_models(dataset_key)
    _model_label_to_key = {label: key for key, label in _model_choices}
    model_sel = mr.Select(label="ML Model", value=_model_choices[0][1], choices=[label for _, label in _model_choices])
    model_key = _model_label_to_key[model_sel.value]

    _CM_LABEL = "Exact  \u2014  CM check \u2192 MR repair"
    _ACM_LABEL = "Approximate  \u2014  ACM check \u2192 AMR repair"
    approach_sel = mr.Select(label="Approach (check \u2192 repair)", value=_ACM_LABEL, choices=[_CM_LABEL, _ACM_LABEL])
    demo_is_acm = approach_sel.value.startswith("Approximate")

    if demo_is_acm:
        acm_threshold = mr.Numeric(label="ACM Threshold", value=5, min=0, max=10)
    else:
        acm_threshold = None

    _method_choices = mc.demo_methods()
    _method_label_to_key = {label: key for key, label in _method_choices}
    _default_method_label = _method_choices[1][1] if len(_method_choices) > 1 else _method_choices[0][1]
    method_sel = mr.Select(label="Imputation Method (repair + baselines)", value=_default_method_label, choices=[label for _, label in _method_choices])
    method_key = _method_label_to_key[method_sel.value]

    start_training = mr.Button(label="Run", style="success")
else:
    x_columns = mr.MultiSelect(label="Input features", value=list(df.columns)[:-1], choices=list(df.columns))
    y_column = mr.Select(label="Target", value=list(df.columns)[-1], choices=list(df.columns))
    _ = mr.Note("<span style='font-size:20px'>**Model Configuration**</span>")
    is_acm = mr.Checkbox(label="Evaluate ACM", value=True)
    verbose = mr.Checkbox(label="Return Samples Requiring Imputation")
    ACM_Threshold = mr.Slider(value=5, min=0, max=10, label="ACM Threshold", step=1)
    eval_metric = mr.Select(label="Evaluation Metric", value="Accuracy", choices=["Accuracy", "RMSE", "MSE"])
    time_limit = mr.Numeric(label="Running Time Limit (Minutes)", value=1, min=1, max=150)
    _ = mr.Note("<span style='font-size:20px'>**Additional Settings**</span>")
    cleaning_function = mr.File(label="Provide Imputation Function (Default Mean)", max_file_size="1MB")
    start_training = mr.Button(label="Start Training", style="success")
    output_dir = mr.OutputDir()
'''.strip()

C_VALIDATE = r'''
if not DEMO:
    if x_columns.value is None or len(x_columns.value) == 0 or y_column.value is None:
        mr.Stop()
'''.strip()

C_DIAGNOSTICS = r'''
%matplotlib inline
_rows_missing = int(df.isnull().any(axis=1).sum())
_pct_missing = 100.0 * _rows_missing / max(len(df), 1)
_overview = (
    f"<div style='display:flex;gap:24px;flex-wrap:wrap;color:#374151;font-size:15px;margin-bottom:8px;'>"
    f"<span><b>Rows:</b> {len(df)}</span>"
    f"<span><b>Features:</b> {df.shape[1] - 1}</span>"
    f"<span><b>Rows with missing values:</b> {_rows_missing} ({_pct_missing:.1f}%)</span>"
    f"</div>"
)
mr.Markdown("# Dirty Value Diagnostics")
display(HTML(_overview))
_mvt = missing_values_table(df)
if len(_mvt) > 0:
    display(_mvt)
msno.matrix(df, figsize=(12, 3), fontsize=10)
plt.show()
'''.strip()

C_DEMO_RESULTS = r'''
if DEMO and start_training.clicked:
    mr.Markdown("# MinPrep Results")

    _thr = acm_threshold.value if demo_is_acm else 0
    check = mc.demo_check(dataset_key, model_key, demo_is_acm, _thr)
    _verdict = "Exists" if check['exists'] else "Does not exist"
    _sub = (f"Approximate check (ACM) with threshold = {check['threshold']}"
            if check['name'] == 'ACM' else "Exact certain-model check (CM)")
    metric_cards(
        f"Step 1 \u00b7 Check \u2014 {check['name']} Result: {_verdict}",
        "#2563eb",
        [(_fmt_score(check['score'], check['score_label']), check['score_label']),
         (f"{check['time_s']:.2f}s", "Running Time")],
        subtitle=_sub,
    )

    if check['exists']:
        mr.Markdown(f"**Step 2 \u00b7 Repair \u2014 not needed.** An accurate model exists for **{model_sel.value}** without repairing the data.")
    else:
        repair = mc.demo_repair(dataset_key, model_key, demo_is_acm, method_key)
        metric_cards(
            f"Step 2 \u00b7 Repair \u2014 {repair['full_name']} ({repair['name']})",
            "#7c3aed",
            [(f"{repair['pct_imputed']:.1f}%", "% Samples Imputed"),
             (f"{repair['rows_imputed']}", "Rows Imputed"),
             (f"{repair['time_s']:.1f}s", "Running Time"),
             (_fmt_score(repair['score'], repair['score_label']), repair['score_label'])],
            subtitle=f"Imputation method: {method_sel.value}",
        )

    ac = mc.demo_baseline(dataset_key, model_key, method_key, 'activeclean')
    metric_cards(
        "Baseline \u00b7 ActiveClean",
        "#d97706",
        [(f"{ac['pct_imputed']:.1f}%", "% Samples Imputed"),
         (f"{ac['time_s']:.1f}s", "Running Time"),
         (_fmt_score(ac['score'], ac['score_label']), ac['score_label'])],
    )

    fi = mc.demo_baseline(dataset_key, model_key, method_key, 'full_impute')
    metric_cards(
        "Baseline \u00b7 Full Imputation",
        "#059669",
        [(f"{fi['pct_imputed']:.1f}%", "% Samples Imputed"),
         (f"{fi['time_s']:.1f}s", "Running Time"),
         (_fmt_score(fi['score'], fi['score_label']), fi['score_label'])],
    )

    mr.Confetti()
'''.strip()

C_GUARD = r'''
# In demo mode everything has been rendered above; stop before the live
# run-mode pipeline executes.
if DEMO:
    mr.Stop()
'''.strip()

# ── Run-mode cells (kept verbatim from the original notebook) ───────────────────
C_RUN_COMPUTE = r'''
output_csv = None
missing_data_table = None
cm_passed = None

if start_training.clicked:
    mr.Markdown("# MinPrep Training Results")

    # Step 0: Preprocessing — restrict to selected input features + target,
    # one-hot encode any non-numeric feature columns, split train/test
    X_train, Y_train, X_test, y_test = preprocess_for_training(df, x_columns.value, y_column.value)

    # Step 1: CM check
    # The task is classification or regression
    task_type = 'classification' if eval_metric.value == 'Accuracy' else 'regression'
    cm_results, missing_data_table, cm_passed, _ = certain_clean_main(
        X_train, Y_train, X_test, y_test, task_type=task_type, verbose=verbose.value)

    if cm_passed:
        # CM exists — data is clean, no imputation needed
        output_csv = cm_results
        mr.Markdown("**CM Result: Exists** — No imputation required.")
    else:
        # CM does not exist — run Minimal Repair
        mr.Markdown("**CM Result: Does not Exist** — Running Minimal Repair (MR)...")
        if task_type == 'regression':
            output_csv, missing_data_table = MR_main_regression(X_train, Y_train, X_test, y_test)
        else:
            output_csv, missing_data_table = MR_main(X_train, Y_train, X_test, y_test)

    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_csv).to_csv(
        SHARED_DIR / "certain_prep_results.csv", index=False)
    mr.Confetti()
'''.strip()

C_RUN_CM = r'''
if output_csv is None:
    mr.Stop()
from IPython.display import display, HTML

cm_lookup = cm_results.set_index('Metric')['Value']
cm_result_str = cm_lookup['CM Result']
cm_time = cm_lookup['Running Time (CM)']
score_label = 'Accuracy (CM)' if 'Accuracy (CM)' in cm_lookup.index else 'MSE (CM)'
cm_accuracy = cm_lookup[score_label]

display(HTML(f"<hr style='border:none;border-top:2px solid #e9ecef;margin:24px 0;'><div style='padding:14px 20px;background:#f8f9fa;border-left:5px solid #2563eb;border-radius:0 8px 8px 0;margin-bottom:16px;'><h3 style='margin:0;color:#2c3e50;font-size:20px;font-weight:600;'>CM Result: {cm_result_str}</h3></div>"))

cm_boxes = [
    (round(cm_time, 4),     'CM Running Time (s)'),
    (round(cm_accuracy, 4), score_label),
]
html = "<div style='display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 24px 0;'>"
for data, title in cm_boxes:
    html += f"<div style='border:1px solid #e0e0e0;border-top:3px solid #2563eb;border-radius:8px;padding:18px 24px;text-align:center;min-width:140px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.06);'><div style='font-size:20px;color:#6c757d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;'>{title}</div><div style='font-size:34px;font-weight:700;color:#212529;'>{data}</div></div>"
html += "</div>"
display(HTML(html))
'''.strip()

C_RUN_MR = r'''
if not cm_passed:
    mr_lookup = output_csv.set_index('Metric')['Value']
    display(HTML(f"<hr style='border:none;border-top:2px solid #e9ecef;margin:24px 0;'><div style='padding:14px 20px;background:#f8f9fa;border-left:5px solid #7c3aed;border-radius:0 8px 8px 0;margin-bottom:16px;'><h3 style='margin:0;color:#2c3e50;font-size:20px;font-weight:600;'>Minimal Imputation Result</h3></div>"))

    mr_score_label = 'Accuracy (MR)' if 'Accuracy (MR)' in mr_lookup.index else 'MSE (MR)'
    minprep_boxes = [
        (mr_lookup['% Repaired (MR)'],          '% Sample Imputed'),
        (round(mr_lookup['Running Time (MR)'], 4), 'Total Running Time (s)'),
        (round(mr_lookup[mr_score_label], 4),     mr_score_label),
    ]
    html = "<div style='display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 24px 0;'>"
    for data, title in minprep_boxes:
        html += f"<div style='border:1px solid #e0e0e0;border-top:3px solid #7c3aed;border-radius:8px;padding:18px 24px;text-align:center;min-width:140px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.06);'><div style='font-size:20px;color:#6c757d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;'>{title}</div><div style='font-size:34px;font-weight:700;color:#212529;'>{data}</div></div>"
    html += "</div>"
    display(HTML(html))
'''.strip()

C_RUN_AC = r'''
display(HTML(f"<hr style='border:none;border-top:2px solid #e9ecef;margin:24px 0;'><div style='padding:14px 20px;background:#f8f9fa;border-left:5px solid #d97706;border-radius:0 8px 8px 0;margin-bottom:16px;'><h3 style='margin:0;color:#2c3e50;font-size:20px;font-weight:600;'>AC Result</h3></div>"))

ac_boxes = [
    ('NA', '% Sample Imputed'),
    ('NA', 'Total Running Time (s)'),
    ('NA', 'Test Accuracy (%)'),
]
html = "<div style='display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 24px 0;'>"
for data, title in ac_boxes:
    html += f"<div style='border:1px solid #e0e0e0;border-top:3px solid #d97706;border-radius:8px;padding:18px 24px;text-align:center;min-width:140px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.06);'><div style='font-size:20px;color:#6c757d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;'>{title}</div><div style='font-size:34px;font-weight:700;color:#212529;'>{data}</div></div>"
html += "</div>"
display(HTML(html))
'''.strip()

C_RUN_FI = r'''
display(HTML(f"<hr style='border:none;border-top:2px solid #e9ecef;margin:24px 0;'><div style='padding:14px 20px;background:#f8f9fa;border-left:5px solid #059669;border-radius:0 8px 8px 0;margin-bottom:16px;'><h3 style='margin:0;color:#2c3e50;font-size:20px;font-weight:600;'>Full Imputation Result</h3></div>"))

fi_boxes = [
    ('NA', '% Sample Imputed'),
    ('NA', 'Total Running Time (s)'),
    ('NA', 'Test Accuracy (%)'),
]
html = "<div style='display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 24px 0;'>"
for data, title in fi_boxes:
    html += f"<div style='border:1px solid #e0e0e0;border-top:3px solid #059669;border-radius:8px;padding:18px 24px;text-align:center;min-width:140px;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.06);'><div style='font-size:20px;color:#6c757d;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;'>{title}</div><div style='font-size:34px;font-weight:700;color:#212529;'>{data}</div></div>"
html += "</div>"
display(HTML(html))
'''.strip()

C_RUN_TABLE = r'''
mr.Markdown(" ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
mr.Markdown("# Minimal Imputation")
if missing_data_table is not None and len(missing_data_table) > 0:
    display(missing_data_table.sample(min(5, len(missing_data_table))))
'''.strip()


def main():
    nb = nbf.v4.new_notebook()
    cells = [
        C_IMPORTS, C_IFRAME, C_MODE, C_INPUT, C_SYNC, C_CONFIG, C_VALIDATE,
        C_DEMO_RESULTS, C_DIAGNOSTICS, C_GUARD,
        C_RUN_COMPUTE, C_RUN_CM, C_RUN_MR, C_RUN_AC, C_RUN_FI, C_RUN_TABLE,
    ]
    nb.cells = [nbf.v4.new_code_cell(src) for src in cells]
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, str(NB_PATH))
    print(f"Wrote {NB_PATH} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
