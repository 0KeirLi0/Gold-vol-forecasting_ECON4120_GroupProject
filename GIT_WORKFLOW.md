# Main-only Git workflow

The project uses `main` as its sole working and integration branch. Contributors and AI agents work directly on `main`; do not create or switch to topic branches or open pull requests for project changes.

## Start work

Make sure the worktree is clean and synchronize `main` before editing:

```bash
git switch main
git pull --ff-only origin main
```

Coordinate shared-interface changes with the affected project stage before editing. If another contributor is changing the same files or the worktree is not clean, coordinate rather than overwriting or stashing their work.

## Validate and commit

- Run the applicable tests and checks before committing. For Python changes, run `python -m pytest`.
- Inspect `git status` and the staged diff. Stage explicit paths when unrelated or sensitive files may exist.
- Make small, meaningful commits with imperative conventional subjects, for example `feat(model): add GARCH baseline`.
- Do not commit generated files, raw data, credentials, or local environment files unless the team has confirmed they are safe and required.
- Never rewrite or force-push `main`.

## Push

Push validated commits directly to `main`:

```bash
git push origin main
```

Do not bypass repository protection. If GitHub rejects a direct push, stop and coordinate with the repository maintainer; do not create a topic branch as a workaround.

## Data and analysis

Treat data and model artifacts as part of the project record. Document data provenance, access restrictions, transformations, and reproducibility details. Do not add licensed, private, or otherwise restricted data to Git unless the team has confirmed both permission and an appropriate storage approach. Tie model comparisons and conclusions to their inputs, evaluation evidence, stated selection criteria, and limitations.
