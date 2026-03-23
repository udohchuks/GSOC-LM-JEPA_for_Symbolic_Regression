from data.aif_dataset import parse_equations_csv
metas = parse_equations_csv('data/FeynmanEquations.csv')
print(f"Parsed {len(metas)} equations")
print(metas[0])
print(metas[4])   # I.9.18 — the 9-variable one
