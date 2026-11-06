"""P55 - NIST AI RMF Risk Register Builder: core logic shared by app.py and eval.py."""
import json, os, re
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "rmf_guidance.json")
FUNCTIONS = ["GOVERN", "MAP", "MEASURE", "MANAGE"]
CHARACTERISTICS = ["Valid and reliable", "Safe", "Secure and resilient", "Accountable and transparent",
                   "Explainable and interpretable", "Privacy-enhanced", "Fair with harmful bias managed"]

# ---------- U1: lifecycle stages -> risk points ----------
STAGES = {
    "Plan & Design": ["unclear purpose or scope creep", "missing stakeholder input", "legal or regulatory requirements overlooked"],
    "Collect & Process Data": ["unrepresentative or biased data", "personal data collected without consent", "poor labelling and data quality"],
    "Build & Use Model": ["wrong model or metric choice", "overfitting and poor generalisation", "third-party model or licence risk"],
    "Verify & Validate": ["inadequate testing on real conditions", "no fairness or hallucination test", "results not reproducible"],
    "Deploy & Use": ["over-reliance by users", "prompt injection or misuse", "no human fallback or appeal"],
    "Operate & Monitor": ["data and concept drift", "no incident reporting", "silent degradation of quality"],
    "Retire / Impacted parties": ["no decommissioning plan", "harm to affected communities", "retained data after shutdown"],
}
STAGE_KEYWORDS = {
    "Plan & Design": ["plan", "design", "goal", "purpose", "requirement"],
    "Collect & Process Data": ["data", "dataset", "image", "records", "documents", "collect", "scrape"],
    "Build & Use Model": ["model", "train", "fine-tune", "llm", "classifier", "detect", "predict", "gpt", "rag"],
    "Verify & Validate": ["test", "validate", "evaluate", "accuracy", "benchmark"],
    "Deploy & Use": ["deploy", "app", "chatbot", "whatsapp", "web", "mobile", "users", "customer", "farmer"],
    "Operate & Monitor": ["monitor", "production", "live", "update", "real-time", "continuous"],
    "Retire / Impacted parties": ["retire", "decommission", "community", "citizen", "patient", "farmer", "customer"],
}

def stage_hits(desc):
    d = desc.lower()
    return {s: sorted({k for k in kws if k in d}) for s, kws in STAGE_KEYWORDS.items()}

# ---------- Guardrail (U5): input sanitiser ----------
INJECTION = [r"ignore (all |any )?(previous|prior|above) (instructions|prompts)", r"system prompt",
             r"reveal (your|the) (instructions|prompt)", r"you are now", r"disregard .{0,20}instructions"]
PII = {"email": r"[\w.+-]+@[\w-]+\.[\w.-]+", "phone": r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
       "aadhaar": r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)", "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b"}

def sanitize(text, max_chars=3000):
    flags = []
    text = (text or "").strip()
    if len(text) < 20:
        return None, ["too_short"]
    if len(text) > max_chars:
        text = text[:max_chars]; flags.append("truncated")
    if any(re.search(p, text, re.I) for p in INJECTION):
        return None, ["prompt_injection_blocked"]
    for name, pat in PII.items():
        if re.search(pat, text):
            text = re.sub(pat, f"[{name.upper()}_REDACTED]", text); flags.append(f"pii_redacted:{name}")
    return text, flags

# ---------- U3: Sentence-BERT retrieval (TF-IDF fallback) ----------
def load_kb():
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)

class Retriever:
    def __init__(self, kb=None):
        self.kb = kb or load_kb()
        self.docs = [f"{e['title']}. {e['text']}" for e in self.kb]
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.emb = self.model.encode(self.docs, normalize_embeddings=True)
            self.backend, self.threshold = "sentence-bert (all-MiniLM-L6-v2)", 0.30
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(self.docs)
            self.emb = self.vec.transform(self.docs)
            self.model, self.backend, self.threshold = None, "tf-idf fallback", 0.10

    def search(self, query, k=3, kinds=None):
        if self.model is not None:
            sims = (self.emb @ self.model.encode([query], normalize_embeddings=True)[0]).tolist()
        else:
            sims = (self.emb @ self.vec.transform([query]).T).toarray().ravel().tolist()
        order = np.argsort(sims)[::-1]
        out = [(self.kb[i], float(sims[i])) for i in order if kinds is None or self.kb[i]["function"] in kinds]
        return out[:k]

# ---------- U1/U2: prompts + LLM ----------
SYSTEM = "You are an AI risk analyst who writes concise NIST AI RMF style risk registers. Reply with JSON only."
SCHEMA = ('Return a JSON array of 8 objects with keys: "stage" (one of %s), "risk" (one sentence), '
          '"likelihood" (1-5), "impact" (1-5), "characteristic" (one of %s), "mitigation" (one sentence).')
FEW_SHOT = ('Example project: "Resume screening tool that ranks job applicants."\n'
            '[{"stage":"Collect & Process Data","risk":"Historic hiring data encodes gender bias so the ranker disadvantages women.",'
            '"likelihood":4,"impact":5,"characteristic":"Fair with harmful bias managed",'
            '"mitigation":"Audit selection rates by group and rebalance or remove proxy features."}]\n\n')

def build_prompt(desc, mode="few-shot", context=""):
    head = SCHEMA % (list(STAGES), CHARACTERISTICS)
    shots = FEW_SHOT if mode == "few-shot" else ""
    ctx = f"Relevant NIST guidance:\n{context}\n\n" if context else ""
    return f"{head}\n\n{shots}{ctx}Project description:\n\"\"\"{desc}\"\"\"\nJSON:"

def call_llm(prompt, temperature=0.2):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        return None, "no GROQ_API_KEY set"
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"}, timeout=60,
                          json={"model": os.getenv("LLM_MODEL", "llama-3.1-8b-instant"), "temperature": temperature,
                                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, f"LLM error: {e}"

def parse_json(txt):
    try:
        a, b = txt.index("["), txt.rindex("]") + 1
        data = json.loads(txt[a:b])
        return data if isinstance(data, list) else None
    except Exception:
        return None

def valid_row(r):
    try:
        return (r["stage"] in STAGES and r["characteristic"] in CHARACTERISTICS and str(r["risk"]).strip()
                and str(r["mitigation"]).strip() and 1 <= int(r["likelihood"]) <= 5 and 1 <= int(r["impact"]) <= 5)
    except Exception:
        return False

def offline_draft(desc):
    """Rule-based drafter used when no LLM key is configured (keeps the demo alive)."""
    d = desc.lower()
    f = lambda *w: any(x in d for x in w)
    rows = []
    def add(stage, risk, l, i, ch, mit): rows.append(dict(stage=stage, risk=risk, likelihood=l, impact=i, characteristic=ch, mitigation=mit))
    add("Plan & Design", "Intended use and affected users are not clearly scoped, so the system is used outside its design context.", 3, 3, "Accountable and transparent", "Write a scope statement, intended-use limits and a named risk owner before build.")
    if f("personal", "customer", "account", "patient", "student", "face", "aadhaar", "records", "user data"):
        add("Collect & Process Data", "Personal data is collected or sent to third-party services without clear consent or minimisation.", 4, 5, "Privacy-enhanced", "Minimise and redact personal data, record consent and restrict retention.")
    add("Collect & Process Data", "Training or reference data is unrepresentative, so quality differs across user groups and regions.", 4, 4, "Fair with harmful bias managed", "Profile data coverage by group and add targeted data collection.")
    if f("llm", "chatbot", "gpt", "generat", "rag", "assistant"):
        add("Deploy & Use", "The generative model produces confident but false answers that users act on.", 4, 4, "Valid and reliable", "Ground answers in retrieved sources, show citations and measure the error rate.")
        add("Deploy & Use", "Prompt injection or jailbreaks make the system leak data or ignore its rules.", 3, 4, "Secure and resilient", "Filter inputs, isolate system prompts and run red-team tests.")
    if f("image", "camera", "vision", "photo", "detect"):
        add("Verify & Validate", "Accuracy drops on real-world images (lighting, devices, unseen classes) compared with lab test data.", 4, 4, "Valid and reliable", "Test on field images and show confidence with a fallback to an expert.")
    if f("farmer", "health", "patient", "medical", "loan", "bank", "safety", "crop"):
        add("Deploy & Use", "Users over-rely on automated advice in a high-stakes setting and a wrong output causes real harm.", 3, 5, "Safe", "Add human review for high-impact outputs and clear limits of use.")
    add("Build & Use Model", "Third-party models, APIs or datasets change or carry unclear licence and provenance terms.", 3, 3, "Accountable and transparent", "Keep a component inventory, pin versions and review licences.")
    add("Operate & Monitor", "Data drift and silent quality decay go unnoticed after launch.", 4, 3, "Valid and reliable", "Log inputs and outcomes, set drift alerts and review monthly.")
    add("Retire / Impacted parties", "No accountable owner, AI policy or escalation path exists, so incidents and harmed users are not handled.", 3, 4, "Accountable and transparent", "Assign an accountable risk owner, approve an AI policy, and set an escalation path and incident log.")
    return rows[:7] + rows[-1:] if len(rows) > 8 else rows

def band(score): return "High" if score >= 15 else "Medium" if score >= 8 else "Low"

def draft_risks(desc, retriever, mode="few-shot"):
    """Returns (rows, meta). RAG context from SBERT feeds the prompt; rows are then grounded in RMF."""
    ctx_hits = retriever.search(desc, k=4)
    ctx = "\n".join(f"- {e['id']} {e['title']}: {e['text']}" for e, _ in ctx_hits)
    raw, err = call_llm(build_prompt(desc, mode, ctx))
    rows, source = None, "llm"
    if raw is not None:
        rows = parse_json(raw)
    json_ok = rows is not None
    if not json_ok:
        rows, source = offline_draft(desc), "offline-template"
    n_before = len(rows)
    rows = [r for r in rows if valid_row(r)]
    meta = dict(source=source, mode=mode, json_valid=json_ok, rows_proposed=n_before, rows_valid=len(rows), error=err,
                retrieval=retriever.backend)
    return rows, meta

def build_register(rows, retriever):
    out = []
    for n, r in enumerate(rows, 1):
        q = f"{r['risk']} {r['mitigation']} {r['characteristic']}"
        cat = retriever.search(q, k=1, kinds=FUNCTIONS)[0]
        top = retriever.search(q, k=1)[0]
        L, I = int(r["likelihood"]), int(r["impact"])
        out.append({"ID": f"R{n:02d}", "RMF Function": cat[0]["function"], "RMF Ref": cat[0]["id"],
                    "Stage": r["stage"], "Risk": r["risk"], "Likelihood": L, "Impact": I, "Score": L * I, "Rating": band(L * I),
                    "Trust characteristic": r["characteristic"], "Mitigation": r["mitigation"],
                    "Guidance match": f"{top[0]['id']} {top[0]['title']}", "Similarity": round(top[1], 3)})
    df = pd.DataFrame(out)
    return df.sort_values(["RMF Function", "Score"], key=lambda s: s.map({f: i for i, f in enumerate(FUNCTIONS)}) if s.name == "RMF Function" else -s).reset_index(drop=True)

# ---------- U5: measured hallucination / grounding test ----------
def grounding_metrics(df, retriever, meta):
    ids = {e["id"] for e in retriever.kb}
    grounded = ((df["Similarity"] >= retriever.threshold) & df["Guidance match"].str.split().str[0].isin(ids)).mean() if len(df) else 0.0
    return {"risks": int(len(df)), "grounding_rate": round(float(grounded), 3), "similarity_threshold": retriever.threshold,
            "json_valid": bool(meta["json_valid"]), "schema_valid_rate": round(meta["rows_valid"] / max(meta["rows_proposed"], 1), 3),
            "functions_covered": int(df["RMF Function"].nunique()) if len(df) else 0}

# ---------- U4: heatmap image ----------
def heatmap(df, title="Risk heatmap (likelihood x impact)"):
    grid = np.zeros((5, 5)); labels = [[[] for _ in range(5)] for _ in range(5)]
    for _, r in df.iterrows():
        grid[r["Likelihood"] - 1, r["Impact"] - 1] += 1; labels[r["Likelihood"] - 1][r["Impact"] - 1].append(r["ID"])
    bg = np.array([[(l + 1) * (i + 1) for i in range(5)] for l in range(5)])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(bg, origin="lower", cmap="RdYlGn_r", vmin=1, vmax=30, alpha=0.75)
    for l in range(5):
        for i in range(5):
            if labels[l][i]:
                ax.text(i, l, "\n".join(", ".join(labels[l][i][j:j + 2]) for j in range(0, len(labels[l][i]), 2)), ha="center", va="center", fontsize=8, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("Impact"); ax.set_ylabel("Likelihood"); ax.set_title(title)
    fig.tight_layout()
    return fig

def to_csv(df): return df.to_csv(index=False).encode("utf-8")
