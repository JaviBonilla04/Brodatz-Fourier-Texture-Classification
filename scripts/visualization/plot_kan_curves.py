import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # guarda a archivo, sin abrir ventana
import matplotlib.pyplot as plt


def load_curves(path):
    """Devuelve curves[(feature, clase)] = (lista_x, lista_phi) y la lista de descriptores en orden."""
    curves = defaultdict(lambda: ([], []))
    features = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["feature"], row["class"])
            curves[key][0].append(float(row["x_original"]))
            curves[key][1].append(float(row["phi"]))
            if row["feature"] not in features:
                features.append(row["feature"])
    return curves, features


def plot_class(curves, features, target, out_path):
    """Un panel por descriptor, todos con la misma escala vertical para poder compararlos."""
    n = len(features)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    all_phi = [v for feat in features for v in curves[(feat, target)][1]]
    lo, hi = min(all_phi), max(all_phi)
    pad = 0.1 * (hi - lo)

    for idx, feat in enumerate(features):
        ax = axes[idx // cols][idx % cols]
        xs, ys = curves[(feat, target)]
        ax.plot(xs, ys, color="#D85A30", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(feat, fontsize=10)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(labelsize=8)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(f"Aporte de cada descriptor al puntaje de {target}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Guardado:", out_path)


def plot_feature(curves, feature, targets, out_path):
    """La curva de un descriptor para varias clases, en la misma gráfica."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target in targets:
        xs, ys = curves[(feature, target)]
        ax.plot(xs, ys, linewidth=2, label=target)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"{feature} (valor original)")
    ax.set_ylabel("aporte al puntaje de la clase")
    ax.set_title(f"Curvas de {feature}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print("Guardado:", out_path)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]

    # Carpeta de salida: --out CARPETA, o results/figures por defecto
    if "--out" in args:
        pos = args.index("--out")
        folder = args[pos + 1]
        args = args[:pos] + args[pos + 2:]
    else:
        csv_folder = os.path.dirname(os.path.abspath(args[0]))
        folder = os.path.join(os.path.dirname(csv_folder), "figures")
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)

    path = args[0]
    curves, features = load_curves(path)
    model = os.path.splitext(os.path.basename(path))[0].replace("kan_curves_", "")

    if len(args) == 2:
        target = args[1].upper()
        plot_class(curves, features, target, os.path.join(folder, f"curves_{model}_{target}.png"))
    else:
        feature = args[1]
        targets = [t.upper() for t in args[2:]]
        plot_feature(curves, feature, targets, os.path.join(folder, f"curves_{model}_{feature}.png"))


if __name__ == "__main__":
    main()