from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import typer
from rich import print as rprint

from cyclops.adapters.keras_h5 import KerasH5Adapter
from cyclops.adapters.onnx_rt import ONNXAdapter
from cyclops.adapters.base import ModelAdapter
from cyclops.attacks.base import ATTACKS, Attack
from cyclops.config import parse_transform_spec
from cyclops.datasets.folder import load_folder_dataset
from cyclops.metrics import top1_accuracy, asr, risk_score
from cyclops.report import write_markdown_report, write_html_report
from cyclops.utils.image import save_image
from cyclops.utils.seed import set_seed

app = typer.Typer(add_completion=False)


def _load_adapter(framework: str, model: str, task: str, input_size: tuple[int, int]) -> ModelAdapter:
    if framework == "keras":
        return KerasH5Adapter(model, task=task, input_size=input_size)
    if framework == "onnx":
        return ONNXAdapter(model, task=task, input_size=input_size)
    raise typer.BadParameter(f"Unsupported framework: {framework}")


def _parse_eps_tokens(tokens: List[str]) -> List[float]:
    values: List[float] = []
    for t in tokens:
        t = t.strip()
        if "/" in t:
            num, den = t.split("/", 1)
            values.append(float(num) / float(den))
        else:
            values.append(float(t))
    return values


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


@app.callback()
def _main_callback():  # noqa: D401
    """Cyclops Vision CLI."""


@app.command()
def audit(
    model: str = typer.Option(..., help="Path to model file (.h5 or .onnx)"),
    framework: str = typer.Option("keras", help="Framework: keras|onnx"),
    task: str = typer.Option("classify", help="Task type (MVP: classify)"),
    data: str = typer.Option(..., help="Path to folder dataset root"),
    attacks: List[str] = typer.Option(["fgsm", "pgd"], help="Attacks to run"),
    eps: List[str] = typer.Option(["2/255"], help="Epsilon values (floats or fractions like 2/255)"),
    pgd_steps: int = typer.Option(10, help="PGD steps"),
    transforms: List[str] = typer.Option(["resize=224x224,jpeg=95,blur=0"], help="Transform specs"),
    out: str = typer.Option("./out/mvp", help="Output directory"),
    seed: int = typer.Option(0, help="Random seed"),
) -> None:
    """Audit model robustness against attacks under transforms and produce a report."""
    set_seed(seed)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine input size from first transform's resize, or default 224x224
    input_size = (224, 224)
    for spec in transforms:
        if "resize=" in spec:
            size_token = spec.split("resize=", 1)[1].split(",")[0]
            h, w = size_token.split("x")
            input_size = (int(h), int(w))
            break

    adapter = _load_adapter(framework, model, task, input_size)

    x, y_onehot, class_names = load_folder_dataset(data, input_size=input_size)

    # Baseline on clean inputs (no transform)
    logits_clean = adapter.predict(x)
    baseline_acc = top1_accuracy(y_onehot, logits_clean)
    rprint(f"Baseline accuracy: {baseline_acc:.3f}")

    # Prepare attacks
    selected_attacks: list[Attack] = []
    for name in attacks:
        atk = ATTACKS.get(name)
        if atk is None:
            rprint(f"[yellow]Unknown attack skipped:[/] {name}")
            continue
        if atk.requires_gradients and framework != "keras":
            rprint(f"[yellow]Skipping gradient-based attack for non-gradient adapter:[/] {name}")
            continue
        selected_attacks.append(atk)

    eps_values = _parse_eps_tokens(eps)

    # Run per transform configuration
    coverage: dict[str, str] = {}
    results: dict[str, dict[str, float]] = {}
    gallery: list[dict[str, str | float]] = []

    for spec in transforms:
        t = parse_transform_spec(spec)
        x_t = t(x.copy())
        logits_t = adapter.predict(x_t)
        acc_t = top1_accuracy(y_onehot, logits_t)
        rprint(f"Transform '{spec}' baseline acc: {acc_t:.3f}")

        for atk in selected_attacks:
            best_acc = 1.0
            best_asr = 0.0
            best_eps = None
            best_adv = None
            best_logits_adv = None
            for e in eps_values:
                kwargs = {"eps": e}
                if atk.name == "pgd":
                    kwargs.update({"steps": pgd_steps})
                x_adv = atk.run(adapter, x_t, y_onehot, **kwargs)
                logits_adv = adapter.predict(x_adv)
                acc_adv = top1_accuracy(y_onehot, logits_adv)
                asr_val = asr(logits_t, logits_adv)
                if acc_adv < best_acc:
                    best_acc = acc_adv
                    best_asr = asr_val
                    best_eps = e
                    best_adv = x_adv
                    best_logits_adv = logits_adv
            results[f"{spec}:{atk.name}"] = {"acc": best_acc, "asr": best_asr}
            coverage[f"{atk.name} @ {spec}"] = "done"

            # Save samples and record probs if we found a best_adv
            if best_adv is not None and best_logits_adv is not None and best_eps is not None:
                img_dir = out_dir / "images" / f"{atk.name}_{best_eps:.5f}".replace("/", "-")
                img_dir.mkdir(parents=True, exist_ok=True)
                probs_clean = _softmax(logits_t)
                probs_adv = _softmax(best_logits_adv)
                clean_idx = np.argmax(logits_t, axis=1)
                adv_idx = np.argmax(best_logits_adv, axis=1)
                changed = np.where(clean_idx != adv_idx)[0]
                # take up to 6 changed examples; if none changed, take first 3
                pick = changed[:6] if changed.size > 0 else np.arange(min(3, x_t.shape[0]))
                for i in pick:
                    clean_path = img_dir / f"{spec}_sample_{i}_clean.png"
                    adv_path = img_dir / f"{spec}_sample_{i}_adv.png"
                    save_image(clean_path, x_t[i])
                    save_image(adv_path, best_adv[i])
                    gallery.append({
                        "attack": atk.name,
                        "transform": spec,
                        "eps": float(best_eps),
                        "clean_path": str(clean_path),
                        "adv_path": str(adv_path),
                        "clean_top1": int(clean_idx[i]),
                        "adv_top1": int(adv_idx[i]),
                        "clean_prob": float(probs_clean[i, clean_idx[i]]),
                        "adv_prob": float(probs_adv[i, adv_idx[i]]),
                    })

    # Aggregate per-attack table (take average across transforms)
    per_attack: dict[str, dict[str, float]] = {}
    for key, row in results.items():
        _, atk_name = key.split(":")
        if atk_name not in per_attack:
            per_attack[atk_name] = {"acc": 0.0, "asr": 0.0, "n": 0.0}
        per_attack[atk_name]["acc"] += row["acc"]
        per_attack[atk_name]["asr"] += row["asr"]
        per_attack[atk_name]["n"] += 1.0
    for atk_name, row in per_attack.items():
        n = max(1.0, row.pop("n", 1.0))
        row["acc"] = row["acc"] / n
        row["asr"] = row["asr"] / n

    # Simple risk score: average accuracy under attack across all entries
    if per_attack:
        avg_acc_under_attack = float(np.mean([row["acc"] for row in per_attack.values()]))
    else:
        avg_acc_under_attack = baseline_acc
    r = risk_score(avg_acc_under_attack)

    summary = {
        "baseline_acc": baseline_acc,
        "risk_score": r,
        "num_images": int(x.shape[0]),
        "transforms": transforms,
        "attacks": [a.name for a in selected_attacks],
        "eps": eps_values,
    }
    tables = {"attacks": per_attack, "coverage": coverage, "gallery": gallery}

    write_markdown_report(out_dir / "report.md", summary, tables)
    write_html_report(out_dir / "report.html", summary, tables)

    rprint(f"[green]Reports written to[/] {out_dir}")
