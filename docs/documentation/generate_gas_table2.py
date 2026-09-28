from archnemesis.Data.gas_data import gas_info, molecule_to_latex

def latex(name):
    """Convert a molecule/isotopologue name into LaTeX."""
    return f"${molecule_to_latex(name)}$"

md = []

# ==========================================================
# Summary table
# ==========================================================

md.append("## Implemented gases\n")

md.append("| ID | Molecule | Mean molecular weight (g mol$^{-1}$) | Number of isotopologues |")
md.append("|---:|:---|---:|---:|")

for mol_id in sorted(gas_info, key=int):

    mol = gas_info[mol_id]

    md.append(
        f"| {mol_id} | "
        f"{latex(mol['name'])} | "
        f"{mol['mmw']:.6f} | "
        f"{len(mol['isotope'])} |"
    )

# ==========================================================
# Detailed isotope tables
# ==========================================================

md.append("\n## Isotopologue definitions\n")

for mol_id in sorted(gas_info, key=int):

    mol = gas_info[mol_id]

    md.append(f"<details>")
    md.append(
        f"<summary>{latex(mol['name'])} &nbsp;&nbsp; (ID = {mol_id})</summary>\n"
    )

    md.append("| Isotopologue ID | Isotopologue | Relative abundance | Molecular weight (g mol$^{-1}$) |")
    md.append("|---:|:---|---:|---:|")

    # ISO_ID = 0
    md.append(
        f"|0|"
        f"{latex(mol['name'])}|"
        f"1.0|"
        f"{mol['mmw']:.6f}|"
    )

    # Individual isotopologues
    for iso_id in sorted(mol["isotope"], key=int):

        iso = mol["isotope"][iso_id]

        iso_name = iso.get("name", "—")

        md.append(
            f"|{iso_id}|"
            f"{latex(iso_name) if iso_name != '—' else '—'}|"
            f"{iso['abun']:.8g}|"
            f"{iso['mass']:.6f}|"
        )

    md.append("\n</details>\n")

markdown = "\n".join(md)

with open("gas_table.txt", "w", encoding="utf-8") as f:
    f.write(markdown)

print("Markdown table written to gas_table.txt")

