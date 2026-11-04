# Git and pull request workflow

Use this workflow for all shared repository changes. The default integration branch is `main`; if the repository later adopts a different default branch or GitHub policy, follow that policy.

## 1. Start a focused branch

- Start from an up-to-date `main` branch and create one branch per issue, task, or coherent change. Do not develop directly on `main`.
- Use a short, descriptive name with a category prefix:
  - `feat/<topic>` — new functionality or analysis
  - `fix/<topic>` — correction to existing behavior or results
  - `data/<topic>` — data acquisition or pipeline work
  - `docs/<topic>` — reports, documentation, or slides
  - `chore/<topic>` — maintenance and project setup
- Keep the branch limited to one reviewable outcome. Coordinate ownership before changing shared data formats, model inputs/outputs, or analysis conventions.

```bash
git switch main
git pull --ff-only
git switch -c feat/describe-the-change
```

## 2. Commit coherent progress

- Make small commits that represent meaningful, reviewable steps. Do not commit generated files, raw data, credentials, or local environment files unless the project explicitly requires them and the team has confirmed they are safe to share.
- Use an imperative subject with a conventional type, for example:
  - `feat(model): add GARCH baseline`
  - `data(pipeline): document required inputs`
  - `docs(report): summarize model limitations`
- Before committing, inspect `git status` and the staged diff. Stage explicit paths rather than everything when unrelated or sensitive files may exist.
- Never rewrite shared branch history or force-push. If a pushed branch needs correction, add a follow-up commit.

## 3. Open a pull request

- Push the topic branch and open a PR against `main` when the change is ready for team review. Draft PRs are appropriate for early feedback on an incomplete change; mark the PR ready when it is reviewable.
- Explain the purpose, important implementation or analysis choices, evidence or validation performed, limitations, and any handoff needed by another project stage. Link related issues or project artifacts when available.
- Request review from at least one teammate who can assess the affected area. For cross-stage changes, include the teammate responsible for the dependent stage. Authors should not approve their own PR.
- Keep PRs small, respond to review comments, and update the PR when the scope or conclusions change. Resolve conversations before merging.

## 4. Validate and merge

- Run the checks applicable to the changed files and report their results in the PR. Until the repository defines automated checks, state which manual checks or analytical validation were performed.
- Merge only after required reviews and checks pass and any requested changes are addressed. If repository settings specify review counts, checks, or a merge strategy, those settings take precedence.
- Prefer GitHub's **squash merge** for a focused PR unless the team or repository settings call for another strategy. Delete the topic branch after merge when it is no longer needed.

## 5. Keep work synchronized

- For a long-running branch, periodically incorporate the latest `main` using the repository's chosen merge or rebase policy. Do not force-push a branch that others use.
- If conflicts occur, inspect both sides and preserve the intended behavior. Run relevant checks after resolving conflicts; ask the affected teammate when a conflict changes shared data, model, or analysis assumptions.
- After merge, start subsequent work from an updated `main` and use a new topic branch.

## Data and analysis changes

Treat data and model artifacts as part of the review, not just code. Document data provenance, access restrictions, transformations, and reproducibility details. Do not add licensed, private, or otherwise restricted data to Git unless the team has confirmed both permission and an appropriate storage approach. Keep model comparisons and conclusions tied to their inputs, evaluation evidence, and stated selection criteria.
