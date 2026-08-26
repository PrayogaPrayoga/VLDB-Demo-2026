"""MinPrep interactive demo — Streamlit wizard.

A page-based flow for the conference demo. In **demo mode** nothing is computed
live; results are served instantly from ``demo_results/results.json`` (placeholder
numbers for now, real paper numbers later).

Flow:
    Landing (Demo / Run)
      └─ Demo
           ├─ 1. Dataset       (select + data profile)
           ├─ 2. Model         (pick ML model)
           ├─ 3. Threshold      (set error tolerance? -> CM vs ACM)
           ├─ 4. Check          (CM/ACM exists? placeholder numbers)
           └─ 5. Imputation     (MR/AMR — stub, to be built)

Run:
    streamlit run minprep_app.py --server.port 8501 --server.address 127.0.0.1
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import minprep_core as mc

# ──────────────────────────────────────────────────────────────────────────────
# Page config + theming
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MinPrep",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PRIMARY = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#059669"
AMBER = "#d97706"

st.markdown(
    """
    <style>
      /* hide default chrome for a cleaner demo */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      div[data-testid="stToolbar"] {visibility: hidden;}

      .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}

      .mp-brand {font-size: 15px; font-weight: 700; letter-spacing: 2px;
                 color: #2563eb; text-transform: uppercase;}
      .mp-hero-title {font-size: 46px; font-weight: 800; color: #0f172a; margin: 4px 0 6px 0;}
      .mp-hero-sub {font-size: 18px; color: #475569; margin-bottom: 8px;}

      .mp-h1 {font-size: 30px; font-weight: 800; color: #0f172a; margin: 2px 0 2px 0;}
      .mp-sub {font-size: 16px; color: #64748b; margin-bottom: 18px;}

      /* step indicator */
      .mp-steps {display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 22px 0;}
      .mp-step {display:flex; align-items:center; gap:8px; padding:7px 14px;
                border-radius:999px; font-size:13.5px; font-weight:600;
                background:#eef2f7; color:#94a3b8; border:1px solid #e2e8f0;}
      .mp-step.active {background:#2563eb; color:#fff; border-color:#2563eb;}
      .mp-step.done {background:#e0edff; color:#2563eb; border-color:#c7dbff;}
      .mp-step .num {display:inline-flex; align-items:center; justify-content:center;
                     width:20px; height:20px; border-radius:50%; font-size:12px;
                     background:rgba(255,255,255,0.35);}

      /* choice cards on landing */
      .mp-card {border:1px solid #e2e8f0; border-radius:16px; padding:26px;
                background:#fff; box-shadow:0 4px 16px rgba(15,23,42,0.05); height:100%;}
      .mp-card h3 {margin:0 0 6px 0; font-size:22px; color:#0f172a;}
      .mp-card p {color:#64748b; font-size:15px; margin:0 0 8px 0;}
      .mp-badge {display:inline-block; font-size:12px; font-weight:700; padding:3px 10px;
                 border-radius:999px; margin-bottom:10px;}

      /* metric tiles */
      .mp-tile {border:1px solid #e2e8f0; border-top:3px solid #2563eb; border-radius:12px;
                padding:16px 18px; text-align:center; background:#fff;}
      .mp-tile .lbl {font-size:12.5px; color:#64748b; text-transform:uppercase;
                     letter-spacing:.5px; margin-bottom:6px;}
      .mp-tile .val {font-size:26px; font-weight:800; color:#0f172a; line-height:1.1;}

      /* verdict banner */
      .mp-verdict {border-radius:12px; padding:18px 22px; margin:6px 0 4px 0;
                   font-size:20px; font-weight:700;}
      .mp-verdict small {display:block; font-weight:500; font-size:14px; margin-top:4px; opacity:.9;}

      .stButton>button {border-radius:10px; font-weight:600; padding:.55rem 1.1rem;}

      /* comparison table (Scenario 1) */
      .mp-cmp {width:100%; border-collapse:collapse; margin:8px 0 4px 0; font-size:15px;
               border:1px solid #e2e8f0; border-radius:12px; overflow:hidden;}
      .mp-cmp th {text-align:left; padding:11px 16px; color:#64748b; font-size:12px;
                  text-transform:uppercase; letter-spacing:.5px; background:#f8fafc;
                  border-bottom:1px solid #e2e8f0;}
      .mp-cmp th.num, .mp-cmp td.num {text-align:right; font-variant-numeric:tabular-nums;}
      .mp-cmp td {padding:13px 16px; border-bottom:1px solid #eef2f7; color:#0f172a;}
      .mp-cmp tr:last-child td {border-bottom:none;}
      .mp-cmp .method {font-weight:600;}
      .mp-cmp .best {background:#f0fdf4;}
      .mp-cmp .best td {font-weight:700; color:#065f46;}
      .mp-cmp .tag {display:inline-block; font-size:11px; font-weight:700; padding:2px 8px;
                    border-radius:999px; background:#dcfce7; color:#059669; margin-left:8px;
                    vertical-align:middle;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# State + navigation
# ──────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "step": "home",
    "mode": None,
    "dataset_key": None,
    "variant_key": None,
    "model_key": None,
    "use_threshold": False,
    "threshold": 0.1,
    "method_key": None,
    "repair_ran": False,
    "repair_method_ran": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def go(step):
    st.session_state.step = step


def go_imputation():
    """Enter Step 5 fresh (hide stale results until the user runs the repair)."""
    st.session_state.repair_ran = False
    st.session_state.repair_method_ran = None
    st.session_state.step = "imputation"


def reset_all():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


@st.cache_data(show_spinner=False)
def load_df(path):
    return pd.read_csv(path)


def fmt_score(value, label):
    return f"{value * 100:.1f}%" if label == "Accuracy" else f"{value:.3f}"


def fmt_time(t):
    return f"{t:.2f}s" if t < 10 else f"{t:.0f}s"


# ── The Race: animated head-to-head runtime comparison ────────────────────────
GREY = "#94a3b8"
MANUAL_S_PER_ROW = 6  # rough analyst effort per row hand-repaired, for context


def fmt_duration(s):
    """Human-friendly duration for large times / effort estimates."""
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    m = s / 60.0
    if m < 90:
        return f"{m:.0f} min"
    h = m / 60.0
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24.0:.1f} days"


def dnf_label(reason, est_s):
    if reason == "oom":
        return "\u2717 out of memory"
    return f"\u23f3 exceeded budget \u00b7 est ~{fmt_duration(est_s)}"


def race_lane(name, res, accent, score_label, winner=False, tag=None, na=False):
    """Build one lane dict for the race component."""
    if na:
        return {"name": name, "na": True, "note": "\u2014 not applicable (non-convex)",
                "accent": accent}
    finished = bool(res["finished"])
    return {
        "name": name,
        "na": False,
        "time_s": float(res["time_s"]),
        "finished": finished,
        "dnf": None if finished else dnf_label(res.get("dnf_reason"), res["time_s"]),
        "score": fmt_score(res["score"], score_label),
        "accent": accent,
        "winner": winner,
        "tag": tag,
    }


def race_saved(minprep_time, full, ac):
    """Return (saved_value_seconds_or_None, note_text) framed against Full/AC."""
    if full["finished"]:
        value = max(0.0, full["time_s"] - minprep_time)
        note = "compute time vs Full Repair"
    else:
        value = None
        note = "Full Repair never finished \u2014 MinPrep did"
    if ac is not None:
        if ac["finished"]:
            note += (f" \u00b7 saved {fmt_duration(max(0.0, ac['time_s'] - minprep_time))} "
                     "compute time vs ActiveClean")
        else:
            note += " \u00b7 ActiveClean never finished"
    return value, note


def race_effort(mr_rows, full_rows, is_manual):
    if is_manual:
        def h(r):
            return "no cleaning" if r == 0 else f"~{fmt_duration(r * MANUAL_S_PER_ROW)} by hand"
        return (f"Effort \u2014 MinPrep: {mr_rows:,} rows ({h(mr_rows)})   \u00b7   "
                f"Full Repair: {full_rows:,} rows ({h(full_rows)})")

    def r(n):
        return "no repair needed" if n == 0 else f"{n:,} rows repaired"
    return f"Effort \u2014 MinPrep: {r(mr_rows)}   \u00b7   Full Repair: {r(full_rows)}"


_RACE_CSS = """<style>
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a}
.hd{display:flex;justify-content:space-between;align-items:center;margin:2px 2px 12px}
.cap{font-size:12px;color:#64748b}
#replay{border:1px solid #cbd5e1;background:#fff;border-radius:8px;padding:4px 11px;font-size:12px;cursor:pointer;color:#334155}
#replay:hover{background:#f8fafc}
.lane{display:grid;grid-template-columns:200px 1fr 176px;align-items:center;gap:12px;margin:7px 0}
.lab{font-size:13px;font-weight:600;line-height:1.15}
.tag{display:inline-block;font-size:10px;font-weight:600;padding:1px 6px;border-radius:999px;background:#f1f5f9;color:#475569;margin-left:6px;vertical-align:middle}
.track{position:relative;height:22px;border-radius:6px;background:#f1f5f9;overflow:hidden}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:6px}
.fill.run{background-image:linear-gradient(90deg,rgba(255,255,255,0) 0%,rgba(255,255,255,.45) 50%,rgba(255,255,255,0) 100%);background-size:220% 100%;animation:sh 1.1s linear infinite}
.fill.win{box-shadow:0 0 0 2px rgba(5,150,105,.25)}
@keyframes sh{0%{background-position:220% 0}100%{background-position:-220% 0}}
.dnfbar{background:repeating-linear-gradient(45deg,#fee2e2,#fee2e2 7px,#fecaca 7px,#fecaca 14px)!important}
.meta{font-size:12px;color:#475569;text-align:right;font-variant-numeric:tabular-nums}
.chip{display:inline-block;font-size:11px;font-weight:700;padding:1px 6px;border-radius:6px;background:#ecfdf5;color:#065f46;margin-left:6px}
.chk{color:#059669;font-weight:700;margin-left:4px}
.wall{color:#b91c1c;font-weight:600;font-size:11px}
.foot{opacity:0;transition:opacity .45s ease;margin-top:12px;border-top:1px solid #eef2f7;padding-top:9px}
.foot.show{opacity:1}
.saved{font-size:15px;font-weight:700;color:#0f172a}
.saved .sub{font-size:12px;color:#64748b;font-weight:500;margin-left:6px}
.effort{font-size:12px;color:#475569;margin-top:4px}
</style>"""

_RACE_BODY = """
<div class="hd"><span class="cap" id="cap"></span><button id="replay">\u21bb Replay</button></div>
<div id="lanes"></div>
<div id="foot" class="foot">
  <div class="saved">\u23f1 <span id="savedval"></span> <span class="sub" id="savednote"></span></div>
  <div class="effort" id="effort"></div>
</div>
"""

_RACE_JS = """
const L = DATA.lanes, T = DATA.screen_ms;
const fins = L.filter(function(l){return !l.na && l.finished;}).map(function(l){return l.time_s;});
const maxFin = Math.max.apply(null, fins.concat([0.001]));
const root = document.getElementById('lanes');
document.getElementById('cap').textContent = DATA.cap;
document.getElementById('savednote').textContent = DATA.saved_note;
document.getElementById('effort').textContent = DATA.effort;
function fmtClock(v){ return v < 10 ? v.toFixed(1)+'s' : Math.round(v)+'s'; }
function fmtDur(s){ if(s<90) return Math.round(s)+' s'; var m=s/60; if(m<90) return Math.round(m)+' min'; var h=m/60; if(h<48) return h.toFixed(1)+' h'; return (h/24).toFixed(1)+' days'; }
const rows = L.map(function(l){
  const lane=document.createElement('div'); lane.className='lane';
  const lab=document.createElement('div'); lab.className='lab';
  lab.innerHTML = l.name + (l.tag ? ('<span class="tag">'+l.tag+'</span>') : '');
  const track=document.createElement('div'); track.className='track';
  const fill=document.createElement('div'); fill.className='fill';
  if(!l.na){ fill.style.backgroundColor=l.accent; }
  track.appendChild(fill);
  const meta=document.createElement('div'); meta.className='meta';
  lane.appendChild(lab); lane.appendChild(track); lane.appendChild(meta);
  root.appendChild(lane);
  if(l.na){ track.style.visibility='hidden'; meta.innerHTML='<span style="color:#94a3b8">'+l.note+'</span>'; }
  return {l:l, fill:fill, meta:meta, done:false, walled:false};
});
let start=null, raf=null;
function frame(ts){
  if(start===null) start=ts;
  const e = ts-start;
  const realElapsed = (e/T)*maxFin;
  let pending=false;
  rows.forEach(function(r){
    const l=r.l; if(l.na) return;
    if(l.finished){
      const w = Math.min(100, realElapsed/l.time_s*100);
      r.fill.style.width = w+'%';
      if(w>=100){
        if(!r.done){ r.done=true; r.fill.classList.remove('run');
          if(l.winner){ r.fill.style.backgroundColor='#059669'; r.fill.classList.add('win'); }
          r.meta.innerHTML = fmtClock(l.time_s)+'<span class="chk">\u2713</span><span class="chip">'+l.score+'</span>';
        }
      } else { r.fill.classList.add('run'); r.meta.textContent = fmtClock(Math.min(realElapsed,l.time_s)); pending=true; }
    } else {
      const p = Math.min(88, (e/(0.8*T))*88);
      r.fill.style.width = p+'%';
      if(e >= 0.8*T){
        if(!r.walled){ r.walled=true; r.fill.classList.remove('run'); r.fill.classList.add('dnfbar');
          r.meta.innerHTML = '<span class="wall">'+l.dnf+'</span>'; }
      } else { r.fill.classList.add('run'); r.meta.innerHTML='<span style="color:#94a3b8">running\u2026</span>'; pending=true; }
    }
  });
  if(e < T || pending){ raf=requestAnimationFrame(frame); } else { finishAll(); }
}
function finishAll(){
  document.getElementById('foot').classList.add('show');
  const sv=document.getElementById('savedval');
  if(DATA.saved_value==null){ sv.textContent='Baselines did not finish'; return; }
  const target=DATA.saved_value, dur=750; let s2=null;
  function cu(ts){ if(s2===null)s2=ts; const p=Math.min(1,(ts-s2)/dur); sv.textContent=fmtDur(target*p)+' saved'; if(p<1) requestAnimationFrame(cu); }
  requestAnimationFrame(cu);
}
function run(){
  start=null;
  rows.forEach(function(r){ if(r.l.na) return; r.done=false; r.walled=false; r.fill.className='fill'; r.fill.style.backgroundColor=r.l.accent; r.fill.style.width='0%'; r.meta.textContent=''; });
  document.getElementById('foot').classList.remove('show');
  if(raf) cancelAnimationFrame(raf);
  raf=requestAnimationFrame(frame);
}
document.getElementById('replay').addEventListener('click', run);
run();
"""


def render_race(lanes, saved_value, saved_note, effort, score_label,
                cap="Replaying measured runtimes \u00b7 compressed to ~10 s"):
    payload = {
        "lanes": lanes,
        "screen_ms": 10000,
        "score_label": score_label,
        "saved_value": saved_value,
        "saved_note": saved_note,
        "effort": effort,
        "cap": cap,
    }
    html = (_RACE_CSS + _RACE_BODY
            + "<script>const DATA = " + json.dumps(payload) + ";\n" + _RACE_JS + "</script>")
    components.html(html, height=150 + len(lanes) * 44)


def tile(label, value, accent=PRIMARY):
    st.markdown(
        f"<div class='mp-tile' style='border-top-color:{accent}'>"
        f"<div class='lbl'>{label}</div><div class='val'>{value}</div></div>",
        unsafe_allow_html=True,
    )


DEMO_STEPS_FULL = [
    ("dataset", "Dataset"),
    ("model", "Model"),
    ("threshold", "Threshold"),
    ("check", "Check"),
    ("imputation", "Repair"),
]
# MLP / FT-Transformer skip the CM/ACM steps.
DEMO_STEPS_SHORT = [
    ("dataset", "Dataset"),
    ("model", "Model"),
    ("imputation", "Repair"),
]


def step_indicator(current, supports_check=True):
    steps = DEMO_STEPS_FULL if supports_check else DEMO_STEPS_SHORT
    order = [s for s, _ in steps]
    cur_i = order.index(current) if current in order else -1
    html = "<div class='mp-steps'>"
    for i, (key, label) in enumerate(steps):
        cls = "active" if i == cur_i else ("done" if i < cur_i else "")
        html += f"<div class='mp-step {cls}'><span class='num'>{i+1}</span>{label}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def header(show_startover=True):
    left, right = st.columns([6, 1])
    with left:
        st.markdown("<div class='mp-brand'>MinPrep</div>", unsafe_allow_html=True)
    with right:
        if show_startover:
            st.button("↺ Start over", on_click=reset_all, width='stretch',
                      key="hdr_startover")


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown("<div class='mp-brand'>MinPrep</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='mp-hero-title'>Minimal Data Cleaning for Model Training</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mp-hero-sub'>Determine whether dirty data needs repair before training — "
        "and if so, repair only the minimal subset needed for an accurate model.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            f"<div class='mp-card'>"
            f"<span class='mp-badge' style='background:#e0edff;color:{PRIMARY}'>RECOMMENDED FOR THE DEMO</span>"
            "<h3>🎬 Demo Mode</h3>"
            "<p>A guided walkthrough with instant, pre-computed results. "
            "Pick a dataset and model, choose your accuracy tolerance, and see whether "
            "repair is needed — no waiting.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Enter Demo Mode  →", type="primary", width='stretch', key="enter_demo"):
            st.session_state.mode = "demo"
            go("dataset")
            st.rerun()
    with c2:
        st.markdown(
            f"<div class='mp-card'>"
            f"<span class='mp-badge' style='background:#f1f5f9;color:#475569'>LIVE COMPUTATION</span>"
            "<h3>⚙️ Run Mode</h3>"
            "<p>Actually execute the CM/ACM checks, minimal repair and baselines on real data "
            "to show artifacts and timings. Slower — used off the clock.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Enter Run Mode  →", width='stretch', key="enter_run"):
            st.session_state.mode = "run"
            go("run")
            st.rerun()


def _ensure_variant(dataset_key):
    """Make sure session variant_key is valid for the current dataset."""
    vkeys = [vk for vk, _, _ in mc.demo_variants(dataset_key)]
    if st.session_state.variant_key not in vkeys:
        st.session_state.variant_key = mc.demo_default_variant(dataset_key)


def synthetic_matrix_figure(feature_missing, n_rows=140, seed=0):
    """Build a missingno-style matrix from the stored per-feature missing %s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import missingno as msno
    rng = np.random.default_rng(seed)
    names = [n for n, _ in feature_missing]
    data = {}
    for name, pct in feature_missing:
        col = rng.random(n_rows).astype(float)
        mask = rng.random(n_rows) < (float(pct) / 100.0)
        col[mask] = np.nan
        data[name] = col
    dfm = pd.DataFrame(data, columns=names)
    fig, ax = plt.subplots(figsize=(11, 3))
    msno.matrix(dfm, ax=ax, fontsize=9, sparkline=False)
    return fig, plt


def page_dataset():
    header()
    st.markdown("<div class='mp-h1'>Step 1 · Choose a dataset</div>", unsafe_allow_html=True)
    st.markdown("<div class='mp-sub'>Dirty samples are assumed to be pre-detected. "
                "Review the data profile, then continue.</div>", unsafe_allow_html=True)
    step_indicator("dataset")

    datasets = mc.demo_datasets()  # [(key, display)]
    labels = [d for _, d in datasets]
    keys = [k for k, _ in datasets]
    default_idx = keys.index(st.session_state.dataset_key) if st.session_state.dataset_key in keys else 0

    dcol, vcol = st.columns([2, 1], gap="large")
    with dcol:
        sel_label = st.selectbox("Dataset", labels, index=default_idx, key="dataset_select")
    dataset_key = keys[labels.index(sel_label)]
    st.session_state.dataset_key = dataset_key
    _ensure_variant(dataset_key)

    meta = mc.demo_dataset_meta(dataset_key)
    variants = mc.demo_variants(dataset_key)  # [(vk, label, level)]
    injected = meta["missingness"] == "injected"

    with vcol:
        if injected:
            vk_list = [vk for vk, _, _ in variants]
            vlabels = [f"{lb}" for _, lb, _ in variants]
            vidx = vk_list.index(st.session_state.variant_key) if st.session_state.variant_key in vk_list else 0
            sel_v = st.selectbox("Injected missingness level", vlabels, index=vidx,
                                 key="variant_select")
            st.session_state.variant_key = vk_list[vlabels.index(sel_v)]
        else:
            st.selectbox("Observed missingness", [variants[0][1]], index=0,
                         key="variant_select", disabled=True)
            st.session_state.variant_key = variants[0][0]

    variant_key = st.session_state.variant_key
    vmeta = mc.demo_variant_meta(dataset_key, variant_key)
    stats = vmeta["stats"]
    task = meta["task"].capitalize()

    st.write("")
    cols = st.columns(4)
    with cols[0]:
        tile("Train samples", f"{stats['train_rows']:,}")
    with cols[1]:
        tile("Features", f"{stats['n_features']:,}")
    with cols[2]:
        tile("Task", task, accent=PURPLE)
    with cols[3]:
        tile("Rows with missing", f"{stats['rows_with_missing']:,} ({stats['pct_missing']:.0f}%)",
             accent=AMBER)

    st.caption(f"Missingness type: {meta['kinds']}"
               + ("  ·  injected at the selected level" if injected else "  ·  naturally observed"))

    st.write("")
    feature_missing, features_total = mc.demo_feature_missingness(dataset_key, variant_key)
    shown = len(feature_missing)
    miss_series = pd.Series({n: p for n, p in feature_missing})

    left, right = st.columns([3, 2], gap="large")
    with left:
        cap = f" (top {shown} of {features_total:,})" if features_total > shown else ""
        st.markdown(f"**Missingness by feature (% of values){cap}**")
        st.bar_chart(miss_series, height=280, color=AMBER)
    with right:
        st.markdown("**Per-feature missing rate**")
        tbl = pd.DataFrame({"% Missing": [f"{p:.1f}%" for _, p in feature_missing]},
                           index=[n for n, _ in feature_missing])
        tbl.index.name = "Feature"
        st.table(tbl)

    with st.expander("Show missingness pattern (matrix)"):
        try:
            seed = abs(hash((dataset_key, variant_key))) % (2**32)
            fig, plt = synthetic_matrix_figure(feature_missing, seed=seed)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Representative pattern synthesized from the per-feature missing rates.")
        except Exception as e:
            st.caption(f"(matrix unavailable: {e})")

    st.write("")
    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        st.button("← Back", on_click=go, args=("home",), width='stretch')
    with b2:
        st.button("Continue →", type="primary", on_click=go, args=("model",), width='stretch')


def page_model():
    header()
    _ensure_variant(st.session_state.dataset_key)
    meta = mc.demo_dataset_meta(st.session_state.dataset_key)
    st.markdown("<div class='mp-h1'>Step 2 · Choose the ML model</div>", unsafe_allow_html=True)
    tgt = meta.get("target")
    tgt_txt = f" (target: <code>{tgt}</code>)" if tgt else ""
    st.markdown(f"<div class='mp-sub'>Target function to train on "
                f"<b>{meta['display']}</b>{tgt_txt}.</div>",
                unsafe_allow_html=True)

    variant_key = st.session_state.variant_key
    models = mc.demo_models(st.session_state.dataset_key, variant_key)  # [(key, display)]
    labels = [d for _, d in models]
    keys = [k for k, _ in models]
    default_idx = keys.index(st.session_state.model_key) if st.session_state.model_key in keys else 0
    sel_label = st.selectbox("ML model", labels, index=default_idx, key="model_select")
    model_key = keys[labels.index(sel_label)]
    st.session_state.model_key = model_key

    supports_check = mc.demo_model_supports_check(
        st.session_state.dataset_key, variant_key, model_key)
    step_indicator("model", supports_check)

    if supports_check:
        st.caption("This model supports CM/ACM checking — MinPrep will first test whether "
                   "any repair is needed before training.")
    else:
        st.info("MLP and FT-Transformer don't support CM/ACM checking — MinPrep goes "
                "directly to **minimal repair (AMR)**.")

    st.write("")
    b1, b2, _ = st.columns([1, 1.8, 3.2])
    with b1:
        st.button("← Back", on_click=go, args=("dataset",), width='stretch')
    with b2:
        if supports_check:
            st.button("Continue →", type="primary", on_click=go, args=("threshold",),
                      width='stretch')
        else:
            st.button("Continue to Minimal Repair →", type="primary",
                      on_click=go_imputation, width='stretch')


def page_threshold():
    header()
    st.markdown("<div class='mp-h1'>Step 3 · Accuracy requirement</div>", unsafe_allow_html=True)
    st.markdown("<div class='mp-sub'>Do you require a strictly optimal model, or is a model "
                "within a small error tolerance acceptable?</div>", unsafe_allow_html=True)
    step_indicator("threshold")

    choice = st.radio(
        "Set an error tolerance threshold?",
        [
            "No — require an exact Certain Model (CM)",
            "Yes — allow an Approximately Certain Model (ACM)",
        ],
        index=1 if st.session_state.use_threshold else 0,
        key="threshold_choice",
    )
    use_threshold = choice.startswith("Yes")
    st.session_state.use_threshold = use_threshold

    if use_threshold:
        st.session_state.threshold = st.number_input(
            "Error tolerance (e)", min_value=0.0,
            value=float(st.session_state.threshold), step=0.1, format="%.2f",
            key="threshold_input",
            help="A model is an ACM if, for every possible repair, its training loss is "
                 "within e of the minimal loss on that repair. Larger e is easier to satisfy.",
        )
        st.info(f"MinPrep will check for an **ACM** with tolerance e = "
                f"{st.session_state.threshold:g}.")
    else:
        st.info("MinPrep will check for an **exact CM** (strict optimality across all repairs).")

    st.write("")
    b1, b2, _ = st.columns([1, 1, 4])
    with b1:
        st.button("← Back", on_click=go, args=("model",), width='stretch')
    with b2:
        st.button("Check Necessity of Data Repair →", type="primary",
                  on_click=go, args=("check",), width='stretch')


def page_check():
    header()
    dkey = st.session_state.dataset_key
    vkey = st.session_state.variant_key
    ds = mc.demo_dataset_meta(dkey)
    vmeta = mc.demo_variant_meta(dkey, vkey)
    model_disp = vmeta["models"][st.session_state.model_key]["display"]
    use_thr = st.session_state.use_threshold
    thr = st.session_state.threshold if use_thr else 0

    check = mc.demo_check(dkey, vkey, st.session_state.model_key, use_thr, thr)
    name = check["name"]  # CM or ACM

    st.markdown(f"<div class='mp-h1'>Step 4 · {name} check</div>", unsafe_allow_html=True)
    route = (f"ACM · tolerance e = {check['threshold']:g}" if name == "ACM" else "CM · exact")
    st.markdown(f"<div class='mp-sub'>{ds['display']} · {vmeta['label']} &nbsp;·&nbsp; "
                f"{model_disp} &nbsp;·&nbsp; {route}</div>",
                unsafe_allow_html=True)
    step_indicator("check")

    exists = check["exists"]
    if exists:
        st.markdown(
            f"<div class='mp-verdict' style='background:#dcfce7;color:#065f46'>"
            f"✓ A {name} exists — no repair needed."
            f"<small>An accurate model can be trained directly on the dirty data; "
            f"dirty samples can be safely ignored with a correctness guarantee.</small></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='mp-verdict' style='background:#fef3c7;color:#92400e'>"
            f"✗ No {name} exists — repair is required."
            f"<small>MinPrep will identify the minimal subset of dirty samples to repair "
            f"so an accurate model can be learned.</small></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    slabel = check["score_label"]
    if exists:
        # MinPrep needs no repair; the two baselines still repair, so we compare
        # against them. Baselines use KNN as the representative repair method
        # (matches the paper's Full Repair = KNN-based replacement).
        methods = [k for k, _ in mc.demo_methods()]
        base_method = "knn" if "knn" in methods else methods[0]
        ac = mc.demo_baseline(dkey, vkey, st.session_state.model_key,
                              base_method, "activeclean")
        full = mc.demo_baseline(dkey, vkey, st.session_state.model_key,
                                base_method, "full_impute")

        # ── The Race: MinPrep (no repair) vs baselines that repair anyway ──
        minprep_res = {"time_s": check["time_s"], "finished": True,
                       "score": check["score"]}
        lanes = [
            race_lane(f"MinPrep · {name}", minprep_res, PRIMARY, slabel,
                      winner=True, tag="no repair"),
            race_lane("ActiveClean", ac, PURPLE, slabel),
            race_lane("Full Repair", full, AMBER, slabel),
        ]
        saved_value, saved_note = race_saved(check["time_s"], full, ac)
        render_race(lanes, saved_value, saved_note,
                    race_effort(0, full["rows_imputed"], base_method == "manual"), slabel)
        st.caption("Scenario 1 — MinPrep returns an accurate model with zero repair, while "
                   "ActiveClean and Full Repair spend time repairing unnecessarily.")

        with st.expander("Show detailed comparison table"):
            # A failed baseline (out of memory / over budget) shows only the
            # failure reason — no accuracy or imputation ratio.
            def _crow(label, res, ratio, best=False):
                if not res["finished"]:
                    return (label, "\u2014", dnf_label(res.get("dnf_reason"), res["time_s"]),
                            "\u2014", best, True)
                return (label, fmt_score(res["score"], slabel), fmt_time(res["time_s"]),
                        ratio, best, False)

            rows = [
                (f"MinPrep · {name}", fmt_score(check["score"], slabel),
                 fmt_time(check["time_s"]), "0%", True, False),
                _crow("ActiveClean", ac, f"{ac['pct_imputed']:.1f}%"),
                _crow("Imputing all", full, "100%"),
            ]
            html = ("<table class='mp-cmp'><thead><tr>"
                    f"<th>Method</th><th class='num'>{slabel}</th>"
                    "<th class='num'>Time</th><th class='num'>Imputation ratio</th>"
                    "</tr></thead><tbody>")
            for method, sc, tm, imp, best, failed in rows:
                tag = "<span class='tag'>no repair</span>" if best else ""
                cls = " class='best'" if best else ""
                tmcls = " style='color:#b91c1c'" if failed else ""
                html += (f"<tr{cls}><td class='method'>{method}{tag}</td>"
                         f"<td class='num'>{sc}</td><td class='num'{tmcls}>{tm}</td>"
                         f"<td class='num'>{imp}</td></tr>")
            html += "</tbody></table>"
            st.markdown(html, unsafe_allow_html=True)

        st.write("")
        b1, b2, _ = st.columns([1, 3, 1.5])
        with b1:
            st.button("← Back", on_click=go, args=("threshold",), width='stretch',
                      key="check_back")
        with b2:
            st.button("✓ Found Accurate Model without Data Repair",
                      type="primary", on_click=reset_all, width='stretch',
                      key="check_exit")
    else:
        colt, _ = st.columns([1, 3])
        with colt:
            tile(f"{name} check time", fmt_time(check["time_s"]), accent=PRIMARY)

        st.write("")
        b1, b2, _ = st.columns([1, 1.4, 3.6])
        with b1:
            st.button("← Back", on_click=go, args=("threshold",), width='stretch')
        with b2:
            st.button("Continue to Imputation →", type="primary",
                      on_click=go_imputation, width='stretch')


def _mark_repair_ran():
    st.session_state.repair_ran = True
    st.session_state.repair_method_ran = st.session_state.method_key


def page_imputation():
    header()
    dkey = st.session_state.dataset_key
    vkey = st.session_state.variant_key
    mkey = st.session_state.model_key
    ds = mc.demo_dataset_meta(dkey)
    vmeta = mc.demo_variant_meta(dkey, vkey)
    model = vmeta["models"][mkey]
    model_disp = model["display"]
    slabel = model["score_label"]
    supports_check = mc.demo_model_supports_check(dkey, vkey, mkey)
    supports_ac = mc.demo_model_supports_activeclean(dkey, vkey, mkey)

    # Repair route: convex models follow their CM/ACM choice (CM→MR, ACM→AMR);
    # MLP / FT-Transformer skip the check and default to AMR.
    is_acm = st.session_state.use_threshold if supports_check else True
    repair_short = "AMR" if is_acm else "MR"

    st.markdown("<div class='mp-h1'>Step 5 · Minimal repair</div>", unsafe_allow_html=True)
    if supports_check:
        route = f"no {'ACM' if is_acm else 'CM'} → {repair_short}"
    else:
        route = f"non-convex model → {repair_short}"
    st.markdown(f"<div class='mp-sub'>{ds['display']} · {vmeta['label']} &nbsp;·&nbsp; "
                f"{model_disp} &nbsp;·&nbsp; {route}</div>", unsafe_allow_html=True)
    step_indicator("imputation", supports_check)

    # ── Imputation method selection ────────────────────────────────────────
    methods = mc.demo_methods()  # [(key, label)]
    mlabels = [l for _, l in methods]
    mkeys = [k for k, _ in methods]
    default_idx = mkeys.index(st.session_state.method_key) if st.session_state.method_key in mkeys else 0
    sel = st.selectbox("Imputation method for repair", mlabels, index=default_idx,
                       key="method_select")
    method_key = mkeys[mlabels.index(sel)]
    st.session_state.method_key = method_key

    st.button(f"Run minimal repair ({repair_short}) →", type="primary",
              on_click=_mark_repair_ran, key="run_repair")

    if not st.session_state.repair_ran or st.session_state.repair_method_ran != method_key:
        st.caption("Pick an imputation method and run to see MinPrep's minimal repair "
                   "compared against the baselines.")
        st.write("")
        st.button("← Back", width='stretch', key="imp_back",
                  on_click=go, args=(("check" if supports_check else "model"),))
        return

    # ── Results: MinPrep minimal repair + three baselines ──────────────────
    repair = mc.demo_repair(dkey, vkey, mkey, is_acm, method_key)
    drop = mc.demo_drop_baseline(dkey, vkey, mkey)
    full = mc.demo_baseline(dkey, vkey, mkey, method_key, "full_impute")
    ac = mc.demo_baseline(dkey, vkey, mkey, method_key, "activeclean") if supports_ac else None

    st.write("")

    is_manual = method_key == "manual"

    if is_manual:
        # Manual repair cost is dominated by domain-expert effort, not machine
        # time — so we hide wall-clock time and compare model quality and how
        # much data each approach makes you hand-repair (imputation ratio).
        st.info("**Manual repair** is done by domain experts, so its dominant cost is "
                "human effort — not machine time. We therefore compare model quality and "
                "how much of the data each approach requires you to hand-repair "
                "(imputation ratio), rather than wall-clock time.")

        m_rows = [
            (f"MinPrep · {repair_short}", fmt_score(repair["score"], slabel),
             f"{repair['pct_imputed']:.1f}%", True, False),
            ("Drop all incomplete samples", fmt_score(drop["score"], slabel), "0%", False, False),
        ]
        if full["finished"]:
            m_rows.append(("Imputing all", fmt_score(full["score"], slabel), "100%", False, False))
        else:
            m_rows.append(("Imputing all", "—", "—", False, False))
        if ac is not None:
            if ac["finished"]:
                m_rows.append(("ActiveClean", fmt_score(ac["score"], slabel),
                               f"{ac['pct_imputed']:.1f}%", False, False))
            else:
                m_rows.append(("ActiveClean", "—", "—", False, False))
        else:
            m_rows.append(("ActiveClean", None, None, False, True))

        html = ("<table class='mp-cmp'><thead><tr>"
                f"<th>Method</th><th class='num'>{slabel}</th>"
                "<th class='num'>Imputation ratio</th>"
                "</tr></thead><tbody>")
        for label, sc, imp, best, na in m_rows:
            if na:
                html += (f"<tr><td class='method'>{label}</td>"
                         "<td colspan='2' style='text-align:center;color:#94a3b8'>"
                         "Not available for this model</td></tr>")
                continue
            tag = "<span class='tag'>minimal repair</span>" if best else ""
            cls = " class='best'" if best else ""
            html += (f"<tr{cls}><td class='method'>{label}{tag}</td>"
                     f"<td class='num'>{sc}</td><td class='num'>{imp}</td></tr>")
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)
        st.caption(f"Scenario 2 — with manual repair, MinPrep asks a domain expert to "
                   f"hand-repair only {repair['pct_imputed']:.1f}% of the dirty samples "
                   f"({repair['rows_imputed']:,} rows) versus 100% "
                   f"({full['rows_imputed']:,} rows) for repairing everything, at matching "
                   "model accuracy.")

        st.write("")
        b1, b2, _ = st.columns([1, 3, 1.5])
        with b1:
            st.button("← Back", width='stretch', key="imp_back2",
                      on_click=go, args=(("check" if supports_check else "model"),))
        with b2:
            st.button("✓ Accurate model found with minimal data repair",
                      type="primary", on_click=reset_all, width='stretch', key="imp_exit")
        return

    # ── The Race: MinPrep minimal repair vs three baselines ────────────────
    lanes = [
        race_lane(f"MinPrep · {repair_short}", repair, PRIMARY, slabel,
                  winner=True, tag="minimal repair"),
        race_lane("Drop incomplete", drop, GREY, slabel, tag="\u26a0 no guarantee"),
        race_lane("Full Repair", full, AMBER, slabel),
    ]
    if ac is not None:
        lanes.append(race_lane("ActiveClean", ac, PURPLE, slabel))
    else:
        lanes.append(race_lane("ActiveClean", None, PURPLE, slabel, na=True))
    saved_value, saved_note = race_saved(repair["time_s"], full, ac)
    render_race(lanes, saved_value, saved_note,
                race_effort(repair["rows_imputed"], full["rows_imputed"], is_manual), slabel)
    st.caption(f"Scenario 2 & 3 — MinPrep repairs only the minimal subset "
               f"({repair['pct_imputed']:.1f}% of dirty samples) via {sel}, matching the "
               "accuracy of far more expensive baselines — which may not even finish at scale.")

    with st.expander("Show detailed comparison table"):
        # A failed method (out of memory / over budget) shows only the failure
        # reason — no accuracy or imputation ratio.
        def _row(label, res, ratio, best=False):
            if not res["finished"]:
                return (label, "\u2014", dnf_label(res.get("dnf_reason"), res["time_s"]),
                        "\u2014", best, False, True)
            return (label, fmt_score(res["score"], slabel), fmt_time(res["time_s"]),
                    ratio, best, False, False)

        rows = [
            _row(f"MinPrep · {repair_short}", repair, f"{repair['pct_imputed']:.1f}%", best=True),
            _row("Drop all incomplete samples", drop, "0%"),
            _row("Imputing all", full, "100%"),
        ]
        if ac is not None:
            rows.append(_row("ActiveClean", ac, f"{ac['pct_imputed']:.1f}%"))
        else:
            rows.append(("ActiveClean", None, None, None, False, True, False))

        html = ("<table class='mp-cmp'><thead><tr>"
                f"<th>Method</th><th class='num'>{slabel}</th>"
                "<th class='num'>Time</th><th class='num'>Imputation ratio</th>"
                "</tr></thead><tbody>")
        for label, sc, tm, imp, best, na, failed in rows:
            if na:
                html += (f"<tr><td class='method'>{label}</td>"
                         "<td colspan='3' style='text-align:center;color:#94a3b8'>"
                         "Not available for this model</td></tr>")
                continue
            tag = "<span class='tag'>minimal repair</span>" if best else ""
            cls = " class='best'" if best else ""
            tmcls = " style='color:#b91c1c'" if failed else ""
            html += (f"<tr{cls}><td class='method'>{label}{tag}</td>"
                     f"<td class='num'>{sc}</td><td class='num'{tmcls}>{tm}</td>"
                     f"<td class='num'>{imp}</td></tr>")
        html += "</tbody></table>"
        st.markdown(html, unsafe_allow_html=True)

    st.write("")
    b1, b2, _ = st.columns([1, 3, 1.5])
    with b1:
        st.button("← Back", width='stretch', key="imp_back2",
                  on_click=go, args=(("check" if supports_check else "model"),))
    with b2:
        st.button("✓ Accurate model found with minimal data repair",
                  type="primary", on_click=reset_all, width='stretch', key="imp_exit")


def page_run():
    header()
    st.markdown("<div class='mp-h1'>Run Mode</div>", unsafe_allow_html=True)
    st.markdown("<div class='mp-sub'>Live computation of CM/ACM checks, minimal repair and "
                "baselines on real data.</div>", unsafe_allow_html=True)
    st.info("🚧 **Run mode — to be wired.**  We're building demo mode first; run mode will "
            "execute the real pipeline here (or hand off to the existing notebook app).")
    st.button("← Back to home", on_click=go, args=("home",))


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────
PAGES = {
    "home": page_home,
    "dataset": page_dataset,
    "model": page_model,
    "threshold": page_threshold,
    "check": page_check,
    "imputation": page_imputation,
    "run": page_run,
}
PAGES.get(st.session_state.step, page_home)()
