# NIST AI RMF Risk Register Builder (P55)

Generative AI mini-project (CA) · **Vinit Kale · Roll 55 · LY AIA 3**

Paste a description of an AI project and get a risk register organised under the four NIST AI RMF functions
(**Govern, Map, Measure, Manage**), a likelihood × impact heatmap, and a CSV export.

**Live app:** `<paste Streamlit/HF Spaces URL>`  ·  **Demo video:** `<paste link>`

## Unit coverage
| Unit | What this app does |
|---|---|
| U1 Foundations | NIST lifecycle stages → risk points; zero-shot vs few-shot prompts; cosine-similarity (latent-space) view |
| U2 GANs & LLM | Transformer LLM (Groq-hosted open model) drafts risks; prompt-based learning compared; attention explained in-app |
| U3 GPT/BERT/multimodal | GPT-style LLM + Sentence-BERT (all-MiniLM-L6-v2) retrieval of RMF guidance; heatmap image output |
| U4 Tools | Python + PyTorch (via sentence-transformers); heatmap generation; CSV export |
| U5 Responsible AI | Grounding (hallucination) test, input guardrails + PII redaction, licence note, [MODEL_CARD.md](MODEL_CARD.md) |

## Run locally
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # optional; without it an offline template drafter is used
streamlit run app.py
python eval.py                     # writes results/ (registers, heatmaps, metrics.json)
python build_report.py             # builds the PDF report from results/
```

## Deploy (Streamlit Community Cloud)
Push to a public GitHub repo → share.streamlit.io → New app → `app.py`. In **Settings → Secrets** add:
```toml
GROQ_API_KEY = "..."
LLM_MODEL = "llama-3.1-8b-instant"   # change if Groq retires this model
```
Never commit keys (`.gitignore` already excludes `secrets.toml`).

## Notes
`data/rmf_guidance.json` holds short paraphrases of NIST AI RMF 1.0 categories, trustworthiness characteristics and
NIST AI 600-1 generative-AI risks. Check wording against the official NIST documents before quoting.
