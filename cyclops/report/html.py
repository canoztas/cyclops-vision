from __future__ import annotations

from pathlib import Path
from typing import Any


def write_html_report(path: str | Path, summary: dict[str, Any], tables: dict[str, Any]) -> None:
    p = Path(path)
    html = [
        "<html><head><meta charset='utf-8'><title>Cyclops Vision Report</title>",
        "<style>body{font-family:Arial, sans-serif; margin: 2rem} table{border-collapse:collapse;margin:1rem 0} td,th{border:1px solid #ccc;padding:6px 10px} .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px} .card{border:1px solid #ddd;padding:8px;border-radius:6px} .pair{display:flex;gap:6px} .pair img{width:48%;height:auto;border:1px solid #ccc}</style>",
        "</head><body>",
        "<h1>Cyclops Vision Report</h1>",
        f"<p><b>Baseline accuracy:</b> {summary.get('baseline_acc', 0):.3f}</p>",
        f"<p><b>Risk score:</b> {summary.get('risk_score', 0):.2f}</p>",
        f"<p><b>Images:</b> {summary.get('num_images', 0)}<br><b>Attacks:</b> {', '.join(summary.get('attacks', []))}<br><b>Eps:</b> {', '.join(f'{e:.5f}' for e in summary.get('eps', []))}</p>",
        "<h2>Results by attack</h2>",
        "<table><tr><th>Attack</th><th>Accuracy</th><th>ASR</th></tr>",
    ]
    for attack_name, row in tables.get("attacks", {}).items():
        html.append(f"<tr><td>{attack_name}</td><td>{row.get('acc', 0):.3f}</td><td>{row.get('asr', 0):.3f}</td></tr>")
    html.append("</table>")
    html.append("<h2>Coverage</h2><ul>")
    for key, val in tables.get("coverage", {}).items():
        html.append(f"<li>{key}: {val}</li>")
    html.append("</ul>")

    gallery = tables.get("gallery", [])
    if gallery:
        html.append("<h2>Gallery (clean vs adv)</h2><div class='grid'>")
        for item in gallery[:12]:
            html.append(
                "<div class='card'>"
                f"<div><b>{item['attack']}</b> @ {item['transform']} eps={item['eps']:.5f}</div>"
                f"<div>clean_top1={item['clean_top1']} ({item['clean_prob']:.2f}) → adv_top1={item['adv_top1']} ({item['adv_prob']:.2f})</div>"
                f"<div class='pair'><img src='{item['clean_path']}' alt='clean'><img src='{item['adv_path']}' alt='adv'></div>"
                "</div>"
            )
        html.append("</div>")

    html.append("</body></html>")
    p.write_text("".join(html), encoding="utf-8")
