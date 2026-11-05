# Shared AI project memory

This folder is the shared handoff space for AI agents and human collaborators. Read this file and the latest entries in [`SESSION_LOG.md`](SESSION_LOG.md) before making changes that depend on project history or decisions.

## How to collaborate through the log

- Append a dated entry to `SESSION_LOG.md` after a meaningful project decision, cross-stage change, completed handoff, or discovery that affects future work. Do not log every command or routine edit.
- Do not rewrite or remove earlier entries to make history look different. If a decision changes, add a new entry that states what changed and why.
- Identify whether a statement is a confirmed team decision, an observed technical fact, a proposal, or an unresolved question. Attribute decisions and evidence; never present a proposal as an agreed decision.
- Keep entries concise but useful to the next person: context, decision or result, affected artifacts, limitations, and the next owner/action when known.
- Check the current files and Git history before relying on an old log entry. The log is a handoff aid, not a substitute for the implementation, README, or Git history.
- Do not store credentials, private personal information, restricted raw data, or sensitive provider exports here. Record only safe metadata and where authorized data can be obtained.
- Keep project status in existing project artifacts. Use this log for dated context and handoffs, not as a duplicate mutable status board.

## Entry format

Add new entries at the top of `SESSION_LOG.md`, newest first:

```text
## YYYY-MM-DD — Short subject
- Author: person/team or AI agent
- Type: decision | implementation | finding | handoff | open question
- Context: ...
- Outcome: ...
- Artifacts: ...
- Next: ... (if applicable)
```
