# Raw benchmark results

Store curated, redacted JSONL result records here. Each record must satisfy `schema.json` and link to an experiment record.

Rules:

- include every trial, including failures and regressions;
- never include prompt text, generated text, absolute user paths, credentials, chat history, serials, or platform UUIDs;
- identify the checkpoint by manifest digest/revision, not a private path;
- report median and dispersion; never replace trials with a best value;
- target-hardware records must state the exact chip and GPU-core count;
- `invalid/` contains explicitly marked harness failures and must never be cited as performance evidence;
- checkpoint/runtime manifests are content-addressed supporting evidence, not benchmark results.
