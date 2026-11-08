# Evaluation data (HotpotQA subset)

Planned dataset: **HotpotQA distractor subset** (multi-hop QA), CC BY-SA 4.0 — verify terms before redistribution.

## Suggested MVP subset
- 20–50 questions with their context paragraphs.
- Convert context to `.md/.txt` files under `data/raw/` (one doc per context group or per paragraph set).
- Create `data/eval/golden.jsonl` entries like:

```json
{"id":"hotpotq_...","question":"...","expected_answer":"...","supporting_facts":["doc_or_chunk_hint", "..."]}
```

Keep only the subset needed for evaluation; do not commit full HotpotQA dumps. Add a reproducible prep script later if downloads are approved.
