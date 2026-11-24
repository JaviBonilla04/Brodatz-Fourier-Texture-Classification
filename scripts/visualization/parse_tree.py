import re,sys

r"""
PREVIEW
python parse_tree.py ..\..\results\models\decision_tree_global.yml 3
python parse_tree.py ..\..\results\models\decision_tree_regional.yml 3
python parse_tree.py ..\..\results\models\decision_tree_combined.yml 3

EXECUTE
python parse_tree.py ..\..\results\models\decision_tree_global.yml all > ..\..\results\rules\decision_tree_global.txt
python parse_tree.py ..\..\results\models\decision_tree_regional.yml all > ..\..\results\rules\decision_tree_regional.txt
python parse_tree.py ..\..\results\models\decision_tree_combined.yml all > ..\..\results\rules\decision_tree_combined.txt
"""

REGIONS = ["G", "R1", "R2", "R3"]
FEATURES = ["entropy", "concentration", "mean_radial_freq", "std_radial_freq"]
ALL_NAMES = [f"{region}_{feature}" for region in REGIONS for feature in FEATURES]

# Columnas que usa cada árbol, según cuántas variables tiene
NAMES_BY_VAR_COUNT = {
    4: ALL_NAMES[0:4],    # global
    12: ALL_NAMES[4:16],  # regional
    16: ALL_NAMES,        # combinado
}

# Índice de clase -> nombre. D14 no existe en el dataset.
PRESENT_CLASSES = [i for i in range(1, 113) if i != 14]

def class_name(idx):
    return f"D{PRESENT_CLASSES[idx]}"

SPLIT_PATTERN = re.compile(r"var:(\d+),\s*quality:([^,]+),\s*le:([^ }]+)")

def read_nodes(path):
    """Devuelve (lista de nodos en preorden, nombres de los descriptores)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    var_count = int(re.search(r"var_count: (\d+)", raw).group(1))
    names = NAMES_BY_VAR_COUNT[var_count]

    blocks = raw.split("nodes:")[1].split("\n      -\n")[1:]
    nodes = []
    for block in blocks:
        depth = int(re.search(r"depth: (\d+)", block).group(1))
        value = float(re.search(r"value: ([\-\d.e]+)", block).group(1))
        split = SPLIT_PATTERN.search(block)

        nodes.append({
            "depth": depth,
            "value": int(value),
            # (índice del descriptor, calidad del corte, umbral)
            "split": (int(split.group(1)), float(split.group(2)), float(split.group(3))) if split else None,
        })
    return nodes, names


def build_tree(nodes):
    """OpenCV guarda los nodos en preorden: nodo, subárbol izquierdo, subárbol derecho."""
    position = 0

    def build():
        nonlocal position
        node = nodes[position]
        position += 1
        if node["split"]:
            node["left"] = build()    # caso "<= umbral"
            node["right"] = build()   # caso "> umbral"
        return node

    root = build()
    assert position == len(nodes), "No se leyeron todos los nodos"
    return root

def print_summary(nodes, names):
    n_leaves = sum(1 for n in nodes if not n["split"])
    max_depth = max(n["depth"] for n in nodes)
    print(f"Nodos: {len(nodes)}   Hojas: {n_leaves}   Profundidad máxima: {max_depth}")

    # Importancia = suma de la calidad de todos los cortes que usan cada descriptor
    importance = {}
    for n in nodes:
        if n["split"]:
            var, quality, _ = n["split"]
            importance[var] = importance.get(var, 0.0) + quality
    total = sum(importance.values())

    print("\nImportancia de cada descriptor:")
    for var, value in sorted(importance.items(), key=lambda item: -item[1]):
        print(f"  {names[var]:24s} {100 * value / total:5.1f} %")


def print_levels(node, names, max_level, level=0):
    indent = "  " * level

    if not node["split"]:
        print(f"{indent}hoja: {class_name(node['value'])}")
        return
    if level >= max_level:
        print(f"{indent}... (subárbol)")
        return

    var, _, threshold = node["split"]
    print(f"{indent}si {names[var]} <= {threshold:.4f}:")
    print_levels(node["left"], names, max_level, level + 1)
    print(f"{indent}si no:")
    print_levels(node["right"], names, max_level, level + 1)


def print_rules_for_class(root, names, target):
    paths = []

    def walk(node, conditions):
        if node["split"]:
            var, _, threshold = node["split"]
            walk(node["left"], conditions + [f"{names[var]} <= {threshold:.4f}"])
            walk(node["right"], conditions + [f"{names[var]} > {threshold:.4f}"])
        elif class_name(node["value"]) == target:
            paths.append(conditions)

    walk(root, [])
    paths.sort(key=len)

    print(f"\n{len(paths)} reglas terminan en {target} (de la más corta a la más larga):")
    for path in paths:
        print(f"  [{len(path)} condiciones] " + " Y ".join(path))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    option = sys.argv[2] if len(sys.argv) > 2 else "3"

    nodes, names = read_nodes(path)
    root = build_tree(nodes)
    print_summary(nodes, names)

    if option.upper().startswith("D"):
        print_rules_for_class(root, names, option.upper())
    else:
        max_level = 10**9 if option == "all" else int(option)
        print(f"\nReglas ({'todos los niveles' if option == 'all' else f'primeros {max_level} niveles'}):")
        print_levels(root, names, max_level)


if __name__ == "__main__":
    main()