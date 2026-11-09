"""Regenerate every image in docs/img from real runs (results/*.json from `python -m swpower all`,
plus direct simulations for the time series). Terminal images are the CLI's real stdout."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swpower import data  # noqa: E402
from swpower.analysis import endurance  # noqa: E402
from swpower.studies import FRONT_ENDS  # noqa: E402
from swpower.transient import FrontEnd, simulate, waveform_from_dict  # noqa: E402

IMG = ROOT / "docs" / "img"
RES = ROOT / "results"
C = {"blue": "#2a6fb0", "orange": "#d9822b", "green": "#3a8f5c", "red": "#c0392b", "grey": "#7f8c8d",
     "purple": "#7d5ba6", "teal": "#1f9e9e"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def j(name):
    return json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / name)
    plt.close(fig)
    print("wrote", name)


def fig_budget():
    r = j("budget")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    prios = r["by_priority_sw_w"]
    labels = ["P1 vital", "P2 essential", "P3 deferrable", "P4 comfort"]
    cols = [C["red"], C["orange"], C["blue"], C["grey"]]
    climates = ["C2_cold", "C1_basic_cold", "temperate", "A2_basic_hot", "A1_hot_dry"]
    x = np.arange(len(climates))
    bottom = np.zeros(len(climates))
    for k, lab, col in zip(["1", "2", "3", "4"], labels, cols):
        v = np.full(len(climates), prios[k])
        a.bar(x, v, bottom=bottom, color=col, label=lab)
        bottom += v
    ecu = np.array([r["thermal"][c]["ecu_w"] for c in climates])
    a.bar(x, ecu, bottom=bottom, color=C["teal"], label="cooling unit (P2)")
    a.set_xticks(x, [c.replace("_", "\n") for c in climates])
    a.set_ylabel("W at 28 V (with growth allowance)")
    a.set_title("Silent-watch load by priority and climate")
    a.legend(fontsize=7, loc="upper left")
    th = r["thermal"]
    t = [th[c]["t_amb"] for c in climates]
    b.plot(t, [th[c]["t_uncooled"] for c in climates], "o-", color=C["red"], label="compartment, no cooling")
    b.axhline(data.zone().t_limit_c, color="k", ls="--", lw=1, label="limit 40 degC (SR-40)")
    b2 = b.twinx()
    b2.bar(t, [th[c]["q_cool_w"] for c in climates], width=4, alpha=0.35, color=C["teal"], label="cooling needed")
    b2.set_ylabel("cooling needed, W")
    b.set_xlabel("ambient air, degC")
    b.set_ylabel("degC")
    b.set_title(f"Thermal budget: {r['equipment_heat_w']:.0f} W equipment heat + 4 crew")
    b.legend(fontsize=7, loc="upper left")
    save(fig, "budget_thermal.png")


def fig_battery():
    v = j("battery_validation")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    p = v["product"]
    a.loglog([x["current_a"] for x in p], [x["product_min"] for x in p], "o", color=C["grey"],
             label="qualified 6T AGM product (data sheet)")
    a.loglog([x["current_a"] for x in p], [x["model_min"] for x in p], "-", color=C["blue"],
             label="model: spec-minimum block")
    a.axvspan(20, 140, color=C["green"], alpha=0.08, label="silent-watch rates")
    a.set_xlabel("constant discharge current per block, A")
    a.set_ylabel("minutes to 10.5 V at 25 degC")
    a.set_title("Battery model vs independent product data")
    a.legend(fontsize=7)
    rs = j("reserve")
    temps = [25.0, 0.0, -18.0, -32.0, -46.0]
    for i, (lab, d) in enumerate(rs.items()):
        ys = [d[str(t)] if d[str(t)] is not None else np.nan for t in temps]
        b.plot(temps, ys, "o-", label=lab, color=list(C.values())[i])
        for t, y in zip(temps, ys):
            if np.isnan(y):
                b.plot(t, 1.05, "x", color=list(C.values())[i])
    b.set_ylim(-0.05, 1.1)
    b.set_xlabel("battery temperature, degC")
    b.set_ylabel("lowest SOC that still starts the engine")
    b.set_title("Engine-start reserve: 3 x 10 s at 600 A, bus >= 16 V\n(x = cannot start even when full)")
    b.legend(fontsize=7)
    save(fig, "battery_reserve.png")


def fig_sizing():
    s = j("sizing")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, cl, title in zip(axes, ["A2_basic_hot", "C1_basic_cold"], ["+43 degC (A2 basic hot)", "-32 degC (C1 basic cold)"]):
        for arch, col, mult, lab in [("A1_split_agm", C["orange"], 2, "6T AGM blocks (2 per string)"),
                                     ("A2_split_li", C["blue"], 1, "Li-ion 6T blocks")]:
            rows = s[arch][cl]
            n = [r["np"] * mult for r in rows]
            num = lambda v: np.nan if v == "inf" else v  # noqa: E731
            ax.plot(n, [num(r["essential_h"]) for r in rows], "o-", color=col, label=f"{lab}: essential P1-P2")
            ax.plot(n, [num(r["full_h"]) for r in rows], "s--", color=col, alpha=0.5, label=f"{lab}: full suite")
        ax.axhline(4, color="k", ls="--", lw=1)
        ax.axhline(12, color="k", ls=":", lw=1)
        ax.text(1, 4.2, "threshold 4 h", fontsize=7)
        ax.text(1, 12.2, "objective 12 h", fontsize=7)
        ax.set_xlabel("mission-bank blocks")
        ax.set_title(title)
        ax.set_ylim(0, 14)
    axes[0].set_ylabel("silent-watch hours, engine and APU off")
    axes[0].legend(fontsize=6.5, loc="upper left")
    save(fig, "sizing.png")


def fig_timeline():
    spec = yaml.safe_load((ROOT / "data" / "studies.yaml").read_text(encoding="utf-8"))["alternatives"]
    fig, axes = plt.subplots(4, 2, figsize=(10, 6.4), sharex="col")
    for col, (key, climate, allow) in enumerate([("A2", "A2_basic_hot", False), ("A5", "A2_basic_hot", True)]):
        sp = spec[key]
        r = endurance(sp["arch"], climate, sp["np"], sp["apu"] if allow else "", hours=12, dt=10, silent_req=not allow)
        res = r["result"]
        t = res.t_h - 0.25
        ax = axes[0, col]
        ax.plot(t, res.soc_m, color=C["blue"], label="mission bank SOC")
        ax.plot(t, res.soc_s, color=C["orange"], label="start bank SOC")
        ax.set_ylabel("SOC")
        ax.legend(fontsize=7)
        ax.set_title((f"A2: {sp['np']} Li-ion 6T, true silence" if not allow else "A5: same bank + 5 kW APU allowed") + ", +43 degC")
        axes[1, col].plot(t, res.p_load, color=C["red"], label="mission-bus load")
        axes[1, col].plot(t, res.p_src, color=C["green"], label="charging source")
        axes[1, col].set_ylabel("W")
        axes[1, col].legend(fontsize=7)
        axes[2, col].step(t, res.level, where="post", color=C["purple"])
        axes[2, col].plot(t, np.where(res.lvd, 4, np.nan), "k.", ms=2)
        axes[2, col].set_yticks([0, 1, 2, 3, 4], ["all", "-P4", "-P3", "P1 only", "LVD"])
        axes[2, col].set_ylabel("shed level")
        axes[3, col].plot(t, res.t_zone, color=C["teal"], label="compartment air")
        axes[3, col].axhline(40, color="k", ls="--", lw=1)
        axes[3, col].set_ylabel("degC")
        axes[3, col].set_xlabel("hours into the watch")
    save(fig, "watch_timeline.png")


def fig_aoa():
    a = j("aoa")
    fig, (x1, x2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    alts = a["alternatives"]
    keys = list(alts)
    fom = a["score"]["fom"]
    vals = [fom.get(k, 0.0) for k in keys]
    cols = [C["green"] if k in fom else C["grey"] for k in keys]
    x1.barh(range(len(keys)), vals, color=cols)
    x1.set_yticks(range(len(keys)), [f"{k}: {alts[k]['label']}" for k in keys], fontsize=7)
    for i, k in enumerate(keys):
        if k not in fom:
            x1.text(0.01, i, "screened out: " + ", ".join(a["score"]["screened_out"][k]), va="center", fontsize=6.5)
    x1.invert_yaxis()
    x1.set_xlabel("figure of merit (weighted utility)")
    x1.set_title("Trade study (Wasson ch. 32)")
    crit = [c["id"] for c in a["criteria"]]
    surv = [k for k in keys if k in fom]
    w = np.arange(len(crit))
    for i, k in enumerate(surv):
        u = [a["score"]["utility"][k][c] for c in crit]
        x2.bar(w + i * 0.4 - 0.2, u, width=0.4, label=k, color=[C["blue"], C["orange"]][i % 2])
    x2.set_xticks(w, crit, rotation=60, fontsize=7, ha="right")
    x2.set_ylabel("utility 0..1")
    x2.set_title(f"Utilities of the survivors; MC weights: {a['mc_weights']}", fontsize=8)
    x2.legend(fontsize=7)
    save(fig, "aoa.png")


def fig_transients():
    d = yaml.safe_load((ROOT / "data" / "transients.yaml").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, case, fe_name, title, scale, unit in [
            (axes[0], "surge_flat_100v", "v1_tvs36", "v1: 36 V TVS at the input, 1275E surge", 1e3, "ms"),
            (axes[1], "f_surge", "v3_tvs122_2fet", "v3: 1275F surge tail trips the 0.2 s fault timer", 1e3, "ms"),
            (axes[2], "f_surge", "v4_final", "v4: 0.8 s fault timer rides through 1275F surge", 1e3, "ms")]:
        spec = FRONT_ENDS[fe_name]
        fe = FrontEnd(**{k: v for k, v in spec.items() if k in FrontEnd.__dataclass_fields__})
        r = simulate(waveform_from_dict(case, d["cases"][case]), fe)
        k = slice(None, None, max(1, len(r.t) // 4000))
        ax.plot(r.t[k] * scale, r.vs[k], color=C["grey"], lw=1, label="source (open circuit)")
        ax.plot(r.t[k] * scale, r.va[k], color=C["orange"], lw=1, label="input node")
        ax.plot(r.t[k] * scale, r.vb[k], color=C["blue"], lw=1.4, label="DC/DC input")
        ax.set_title(f"{title}\nTVS {r.e_tvs:.1f} J, FET {r.e_fet:.1f} J, Tj {r.tj_peak:.0f} degC", fontsize=8)
        ax.set_xlabel(unit)
        ax.set_ylabel("V")
        ax.legend(fontsize=6.5)
    save(fig, "transients.png")


def fig_sspc_mc():
    s = j("sspc")
    m = j("monte_carlo")
    r = j("rma")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.4))
    mult = [x[0] for x in s["curve"]]
    trip = [x[1] if x[1] not in ("inf", None) else np.nan for x in s["curve"]]
    dmg = [x[2] if x[2] not in ("inf", None) else np.nan for x in s["curve"]]
    a.loglog(np.array(mult) * 10, trip, color=C["blue"], label="SSPC 10 A trip time")
    a.loglog(np.array(mult) * 10, dmg, color=C["red"], label=f"{s['rep_wire_mm2']} mm2 wire reaches 150 degC")
    a.set_xlabel("channel current, A")
    a.set_ylabel("s")
    a.set_title(f"Protection coordination (fault coverage {s['coverage']})", fontsize=8)
    a.legend(fontsize=7)
    b.hist(m["samples"], bins=20, color=C["blue"], alpha=0.8)
    b.axvline(4, color="k", ls="--")
    b.set_xlabel("essential-suite hours, +43 degC")
    b.set_title(f"Monte Carlo, {m['n']} runs: P(>= 4 h) = {m['p_meets_threshold']:.2f}\n"
                f"P5 {m['p05_h']:.1f} h, median {m['p50_h']:.1f} h", fontsize=8)
    imp = r["rbd"]["k = n - 1 (one spare block)"]["importance_12h"]
    c.barh([x[0] for x in imp][::-1], [x[1] for x in imp][::-1], color=C["orange"])
    c.set_xlabel("share of 12 h unreliability")
    c.set_title(f"RBD, recommended growth path (R12 = {r['rbd']['k = n - 1 (one spare block)']['R_12h']:.3f})",
                fontsize=8)
    save(fig, "sspc_mc_rma.png")


def fig_decision_conflict_arc():
    d = j("decision")
    c = j("conflict")
    a = j("arc")
    fig, (x1, x2, x3) = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1.1]})
    labels = [f"{tag}\n{t:+.0f}" for tag, t, g in d["conditions"]]
    rows = d["rows"]
    mat = np.array([r["hours"] for r in rows])
    im = x1.imshow(mat, cmap="RdYlGn", vmin=0, vmax=16, aspect="auto")
    x1.set_xticks(range(len(labels)), labels, fontsize=7)
    x1.set_yticks(range(len(rows)), [f"{r['posture'].split(' (')[0]}, {r['blocks']} Li" for r in rows], fontsize=7)
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            x1.text(k, i, f"{mat[i, k]:.0f}" if mat[i, k] < 24 else ">24", ha="center", va="center", fontsize=7)
    x1.set_title("Silent hours (engine + APU off)\nby crew posture, bank and condition", fontsize=8)
    fig.colorbar(im, ax=x1, fraction=0.03)
    cases = c["cases"]
    w = 0.2
    for i, r in enumerate(c["rows"]):
        x2.bar(np.arange(len(cases)) + (i - 1.5) * w, [r[k] for k in cases], width=w, label=r["bank"])
    x2.axhline(16, color="k", ls="--", lw=1, label="1275 cranking floor 16 V")
    x2.axhline(14.4, color=C["red"], ls=":", lw=1, label="32565C allows 14.4 V")
    x2.set_xticks(range(len(cases)), [k.replace(", ", "\n") for k in cases], fontsize=6.5)
    x2.set_ylim(10, 27)
    x2.set_ylabel("lowest bank voltage while cranking, V")
    x2.legend(fontsize=6, loc="upper left", ncol=2)
    x2.set_title("Spec-minimum start banks while cranking", fontsize=8)
    cl = a["classes"]
    names = list(cl)
    x3.barh(range(len(names)), [cl[n]["tripped"] / cl[n]["n"] for n in names],
            color=[C["green"] if not n.startswith("nuisance") else C["red"] for n in names])
    x3.set_yticks(range(len(names)), [n.replace("nuisance: ", "nuisance\n") + f" (n={cl[n]['n']})" for n in names], fontsize=6.5)
    x3.set_xlim(0, 1.05)
    x3.set_xlabel("share tripped\n(green: should trip; red: must not)")
    for k, n in enumerate(names):
        x3.text(0.02, k, f"{cl[n]['tripped']}/{cl[n]['n']}", va="center", fontsize=7)
    sm, pm = cl["series"]["max_ms"], cl["parallel"]["max_ms"]
    x3.set_title(f"Arc detector, held-out seeds\nseries <= {sm:.0f} ms, parallel <= {pm:.0f} ms", fontsize=8)
    save(fig, "decision_conflict_arc.png")


def term_png(cmd: list[str], name: str, title: str):
    out = subprocess.run([sys.executable, "-m", "swpower", *cmd], cwd=ROOT, capture_output=True, text=True).stdout
    lines = [f"$ python -m swpower {' '.join(cmd)}"] + out.rstrip().splitlines()
    h = 0.17 * len(lines) + 0.4
    fig = plt.figure(figsize=(10, h))
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.01, 1 - 0.12 / h, title, color="#9cdcfe", family="monospace", fontsize=8, va="top")
    fig.text(0.01, 1 - 0.32 / h, "\n".join(lines), color="#d4d4d4", family="monospace", fontsize=7.5, va="top")
    fig.savefig(IMG / name, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    fig_budget()
    fig_battery()
    fig_sizing()
    fig_timeline()
    fig_aoa()
    fig_transients()
    fig_sspc_mc()
    fig_decision_conflict_arc()
    term_png(["budget"], "cli_budget.png", "power and thermal budget (CLI output)")
    term_png(["validate"], "cli_validate.png", "battery model validation (CLI output)")
    term_png(["watch", "--arch", "A2_split_li", "--np", "10", "--climate", "A2_basic_hot", "--hours", "8"],
             "cli_watch.png", "silent watch, recommended bank, +43 degC (CLI output)")


if __name__ == "__main__":
    main()
