# Contributing to GridSentinel

Thanks for your interest! Contributions of all sizes are welcome — bug fixes,
new detectors, new attack scenarios (defensive, synthetic only), docs and tests.

## Ground rules

- This is a **defensive** project. Please read `ETHICS.md` first. Changes that add
  offensive capability against real systems are out of scope.
- Keep everything **synthetic and offline**. No code that reaches real networks,
  devices, or services.

## Getting set up

```bash
git clone https://github.com/Lohit-18/GridSentinel.git
cd GridSentinel
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard]"
pytest -q
```

## Making a change

1. Create a branch: `git checkout -b feature/my-change`.
2. Add or update **tests** in `tests/` for anything you change.
3. Keep functions small and documented; match the existing style.
4. Run `pytest -q` — everything must pass.
5. For a new detector, plug it into `gridsentinel.detect` and add it to the
   comparison in `gridsentinel.eval.experiments` so it shows up in the report.
6. For a new attack family, add it to `gridsentinel.attacks` and
   `ATTACK_TYPES`, and make sure it carries a clear defensive/synthetic purpose.

## Pull requests

- Describe **what** changed and **why**.
- Include before/after numbers if you touched detection quality.
- One focused change per PR is easier to review.

## Reporting bugs / ideas

Open an issue using the templates in `.github/ISSUE_TEMPLATE/`.
