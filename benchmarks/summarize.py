"""Summarize saved application-level runs; plotting is optional."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(root):
    rows = []
    for traffic in ("steady", "burst"):
        files = sorted(root.glob(traffic + "-*.json"))
        for kind in ("dense", "sequence"):
            for mode in ("unbatched", "sync", "pipeline"):
                runs = [r for f in files for r in json.loads(f.read_text())["runs"]
                        if r["kind"] == kind and r["mode"] == mode]
                if not runs: continue
                rows.append(dict(traffic=traffic, kind=kind, mode=mode, repetitions=len(runs),
                    on_time=sum(r["completed"] for r in runs), offered=sum(r["requests"] for r in runs),
                    median_goodput_rps=statistics.median(r["goodput_rps"] for r in runs),
                    median_p99_ms=statistics.median(r["p99_ms"] for r in runs),
                    median_generator_lag_p99_ms=statistics.median(r["generator_lag_p99_ms"] for r in runs),
                    rates=[100 * r["completed"] / r["requests"] for r in runs]))
    return rows


def chart(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=.08, right=.99, bottom=.22, top=.78)
    groups = [(t, k) for t in ("steady", "burst") for k in ("dense", "sequence")]
    x, width = np.arange(len(groups)), .24
    modes = [("unbatched", "One request", "#7891b2"),
             ("sync", "Batching, 1 slot", "#d99837"),
             ("pipeline", "Batching, 2 slots", "#18897c")]
    for i, (mode, label, color) in enumerate(modes):
        selected = [next(r for r in rows if (r["traffic"],r["kind"],r["mode"]) == (*g,mode)) for g in groups]
        values = [100 * r["on_time"] / r["offered"] for r in selected]
        errors = [[v - min(r["rates"]) for r,v in zip(selected,values)],
                  [max(r["rates"]) - v for r,v in zip(selected,values)]]
        bars = ax.bar(x+(i-1)*width, values, width, label=label, color=color,
                      yerr=errors, capsize=3, error_kw={"lw":1})
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+2, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x, ["Steady\nDense", "Steady\nSequence", "Burst\nDense", "Burst\nSequence"])
    ax.set_ylim(0,112)
    ax.set_ylabel("Requests completed within 100 ms (%)")
    ax.set_title("Application-level deadline completion on Thunder L40", loc="left", weight="bold", pad=36)
    ax.legend(loc="upper left", bbox_to_anchor=(0,1.12), ncol=3, frameon=False)
    ax.yaxis.grid(True, alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.01, .02, "1,500 requests × 3 seeds per configuration; bars: aggregate rate; whiskers: per-run range.\n"
             "Virtualized GPU; shared-process load generator. This chart does not establish CUDA stream overlap.", fontsize=9)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/gpu"))
    p.add_argument("--plot", type=Path)
    args=p.parse_args()
    rows=summarize(args.root)
    (args.root.parent/"gpu-summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    if args.plot:
        args.plot.parent.mkdir(parents=True,exist_ok=True)
        chart(rows,args.plot)


if __name__ == "__main__": main()
