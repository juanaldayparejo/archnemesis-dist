from archnemesis.Data.gas_data import gas_info, molecule_to_html

html = []

# =============================================================================
# CSS
# =============================================================================

html.append("""
<style>

details {
    margin-bottom: 12px;
}

summary {
    cursor: pointer;
    font-size: 1.05em;
    padding: 4px 0;
}

table {
    border-collapse: collapse;
    margin-top: 10px;
    margin-bottom: 10px;
}

th, td {
    border: 1px solid #d0d0d0;
    padding: 8px 14px;
    vertical-align: middle;
}

th {
    background-color: #f5f5f5;
    text-align: center;
}

td {
    text-align: center;
}

tbody tr:hover {
    background-color: #fafafa;
}

/* Column widths */

th:nth-child(1),
td:nth-child(1) {
    min-width: 130px;
}

th:nth-child(2),
td:nth-child(2) {
    min-width: 260px;
}

th:nth-child(3),
td:nth-child(3) {
    min-width: 170px;
}

th:nth-child(4),
td:nth-child(4) {
    min-width: 190px;
}

/* Left-align isotopologue names */

td:nth-child(2) {
    text-align: left;
}

</style>
""")

# =============================================================================
# One dropdown per gas
# =============================================================================

for mol_id in sorted(gas_info, key=int):

    mol = gas_info[mol_id]

    html.append(f"""
<details>

<summary>{molecule_to_html(mol['name'])} (ID = {mol_id})</summary>

<table>

<thead>
<tr>
<th>Isotopologue ID</th>
<th>Isotopologue</th>
<th>Relative abundance</th>
<th>Molecular weight (g mol<sup>-1</sup>)</th>
</tr>
</thead>

<tbody>
""")

    for iso_id in sorted(mol["isotope"], key=int):

        iso = mol["isotope"][iso_id]

        name = iso.get("name", "—")
        if name != "—":
            name = molecule_to_html(name)

        html.append(f"""
<tr>
<td>{iso_id}</td>
<td>{name}</td>
<td>{iso['abun']:.8g}</td>
<td>{iso['mass']:.6f}</td>
</tr>
""")

    html.append("""
</tbody>

</table>

</details>

""")

# =============================================================================
# Write file
# =============================================================================

with open("gas_table.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html))

print("Written gas_table.html")