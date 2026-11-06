# Model Card — NIST AI RMF Risk Register Builder

**Purpose:** draft a first-pass AI risk register for student/small-team projects. **Not** a compliance or legal tool.

**Models:** (1) LLM via Groq API, default `llama-3.1-8b-instant` (configurable) for drafting; (2) `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0) for retrieval. No model was fine-tuned.
**Fallbacks:** TF-IDF retrieval if Sentence-BERT cannot load; a rule-based template drafter if no LLM key/response.
**Data:** 36 hand-written guidance paraphrases (`data/rmf_guidance.json`); no user data stored.

**Evaluation:** grounding rate (share of risks whose best guidance match passes a similarity threshold), JSON/schema validity, function coverage, 3 guardrail tests. See `results/metrics.json`.

**Limitations:** LLM likelihood/impact scores are opinions, not measurements; the function mapping follows embedding similarity and can be wrong (e.g. governance risks mapped to MAP); only 2 sample projects were evaluated; English only.
**Risks:** over-trust in generated registers; prompt injection (mitigated by filtering, not eliminated); project text is sent to a third-party API when a key is set; PII redaction is regex-based and can miss formats.
**Intended use:** brainstorming, teaching. A human must review every row.
**Licence/IP:** NIST AI RMF is a US-government work; text here is paraphrased. Code: MIT (add LICENSE file).
