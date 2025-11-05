# Cyclops Vision

Modular adversarial-robustness fuzzer for vision models.

MVP supports:
- Adapters: Keras (.h5) white-box; ONNX (predict-only)
- Attacks: FGSM, PGD (L∞)
- Transforms: Resize, JPEG, Blur, Compose
- Metrics: baseline acc, attacked acc, ASR, risk score
- Reporters: Markdown + HTML with tables and image gallery

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Optional tooling:
```bash
pip install pytest pre-commit
pre-commit install
```

## Quickstart

1) Generate a toy model and dataset:
```bash
python scripts/generate_toy.py
```

2) Run an audit:
```bash
cyclops audit \
  --model ./toy_model.h5 --framework keras --task classify \
  --data ./examples/imagenet_like \
  --attacks fgsm --attacks pgd \
  --eps 2/255 --eps 4/255 \
  --pgd-steps 10 \
  --transforms "resize=224x224,jpeg=95,blur=0" \
  --transforms "resize=224x224,jpeg=80,blur=1" \
  --out ./out/mvp
```

Outputs:
- `out/mvp/report.md` and `out/mvp/report.html`
- `out/mvp/images/` with clean vs adv image pairs per attack/eps

## Using real models and images

Download MobileNetV2 and a few real images, then build a small dataset:
```bash
python scripts/fetch_real_assets.py
```
Now run `cyclops audit` against `mobilenetv2_imagenet.h5` and `examples/imagenet_like`.

## CLI reference

```text
cyclops audit [OPTIONS]

Options:
  --model TEXT        Path to model file (.h5 or .onnx) [required]
  --framework TEXT    keras|onnx [default: keras]
  --task TEXT         MVP: classify [default: classify]
  --data TEXT         Folder dataset root [required]
  --attacks TEXT      Repeatable. e.g., --attacks fgsm --attacks pgd
  --eps TEXT          Repeatable. Floats or fractions, e.g., 2/255 4/255
  --pgd-steps INTEGER PGD steps [default: 10]
  --transforms TEXT   Repeatable. e.g., "resize=224x224,jpeg=95,blur=0"
  --out TEXT          Output directory [default: ./out/mvp]
  --seed INTEGER      Random seed [default: 0]
```

Transform spec grammar:
- `resize=HxW` (e.g., `224x224`)
- `jpeg=QUALITY` (e.g., `95`)
- `blur=SIGMA` (e.g., `1`)
- Compose via comma: `resize=224x224,jpeg=80,blur=1`

## Report contents

- Summary: baseline accuracy, risk score, dataset size, attacks, eps
- Results table: per-attack average accuracy and ASR (averaged over transforms)
- Coverage: which (attack, transform) combos ran
- Gallery: clean vs adv thumbnails with top-1 label and probability

## Architecture

```
cyclops/
  adapters/  (KerasH5Adapter, ONNXAdapter)
  attacks/   (FGSM, PGD + registry)
  transforms/(Resize, JPEG, Blur, Compose)
  datasets/  (folder loader)
  metrics/   (accuracy, ASR, risk)
  report/    (Markdown, HTML)
  cli.py     (audit command)
```

- Adding an attack: drop a new file in `cyclops/attacks/` and `register()` it.
- Adding a framework: create an adapter in `cyclops/adapters/` implementing `ModelAdapter`.

## Development

- Format/lint/type-check:
```bash
black . && ruff check . && mypy .
```
- Run tests:
```bash
pytest -q
```

## Roadmap (post-MVP)
- More attacks (e.g., EOT/patch-based)
- PyTorch adapter with gradients
- ONNX gradient approximations / transfer attacks
- Object detection adapters and metrics
- Defenses (TTA, sanitize)
