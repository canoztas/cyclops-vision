from __future__ import annotations

from pathlib import Path
from typing import Any


def write_markdown_report(path: str | Path, summary: dict[str, Any], tables: dict[str, Any]) -> None:
    p = Path(path)
    lines: list[str] = []
    lines.append(f"# Cyclops Vision Report\n\n")
    lines.append("## Summary\n")
    lines.append(f"- Baseline accuracy: **{summary.get('baseline_acc', 0):.3f}**\n")
    lines.append(f"- Risk score: **{summary.get('risk_score', 0):.2f}**\n")
    lines.append(f"- Images: {summary.get('num_images', 0)}\n")
    lines.append(f"- Attacks: {', '.join(summary.get('attacks', []))}\n")
    lines.append(f"- Eps: {', '.join(f'{e:.5f}' for e in summary.get('eps', []))}\n")
    lines.append("\n## Results by attack\n")
    lines.append("| Attack | Accuracy | ASR |\n|---|---:|---:|\n")
    for attack_name, row in tables.get("attacks", {}).items():
        lines.append(f"| {attack_name} | {row.get('acc', 0):.3f} | {row.get('asr', 0):.3f} |\n")

    lines.append("\n## Coverage (attack x transform)\n")
    for key, val in tables.get("coverage", {}).items():
        lines.append(f"- {key}: {val}\n")

    gallery = tables.get("gallery", [])
    if gallery:
        lines.append("\n## Gallery (clean vs adv)\n")
        for item in gallery[:12]:
            lines.append(
                f"- {item['attack']} @ {item['transform']} eps={item['eps']:.5f} "
                f"clean_top1={item['clean_top1']} ({item['clean_prob']:.2f}) → "
                f"adv_top1={item['adv_top1']} ({item['adv_prob']:.2f})\n\n"
            )
            lines.append(f"  ![clean]({item['clean_path']})  ![adv]({item['adv_path']})\n\n")

    p.write_text("".join(lines), encoding="utf-8")
