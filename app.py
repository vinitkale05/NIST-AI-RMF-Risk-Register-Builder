import json, os
import pandas as pd
import streamlit as st
import core

st.set_page_config(page_title="NIST AI RMF Risk Register Builder", page_icon="🛡️", layout="wide")
try:  # keys live in platform secrets, never in the repo
    for k in ("GROQ_API_KEY", "LLM_MODEL"):
        if k in st.secrets: os.environ[k] = st.secrets[k]
except Exception:
    pass

@st.cache_resource(show_spinner="Loading Sentence-BERT and RMF guidance...")
def get_retriever(): return core.Retriever()
retr = get_retriever()
samples = json.load(open("samples/projects.json", encoding="utf-8"))

st.title("🛡️ NIST AI RMF Risk Register Builder")
st.caption("P55 · Generative AI mini-project · Vinit Kale · Roll 55 · LY AIA 3")
live = bool(os.getenv("GROQ_API_KEY"))
st.sidebar.markdown(f"**LLM:** {'Groq API connected' if live else 'no key → offline template drafter'}  \n**Retriever:** {retr.backend}")
mode = st.sidebar.radio("Prompt technique", ["few-shot", "zero-shot"], help="Prompt-based learning (U2). Compare both below.")
compare = st.sidebar.checkbox("Compare zero-shot vs few-shot", value=False, disabled=not live)

choice = st.selectbox("Load a sample project or write your own", ["(write my own)"] + list(samples))
desc = st.text_area("Describe your AI project (purpose, users, data, model, where it runs)", height=150,
                    value="" if choice == "(write my own)" else samples[choice])

t1, t2, t3, t4 = st.tabs(["1 · Lifecycle map", "2 · Risk register", "3 · Heatmap & CSV", "4 · Responsible AI"])
clean, flags = core.sanitize(desc) if desc else (None, ["empty"])
if desc and clean is None:
    st.error(f"Input rejected by guardrail: {flags[0]}. Please describe your project in plain words (20+ characters).")
if flags and clean is not None and any(f.startswith("pii") for f in flags):
    st.warning("Personal data was detected and redacted before processing: " + ", ".join(flags))

with t1:
    st.markdown("**U1 · Where in the AI lifecycle can things go wrong?** Stages come from the NIST lifecycle; keywords found in your text show which stages your description touches.")
    if clean:
        hits = core.stage_hits(clean)
        st.dataframe(pd.DataFrame([{"Stage": s, "Typical risk points": "; ".join(r), "Keywords in your text": ", ".join(hits[s]) or "-"}
                                   for s, r in core.STAGES.items()]), width="stretch", hide_index=True)
    with st.expander("How the LLM and retrieval work (U2 attention · latent-space view)"):
        st.markdown("The LLM is a Transformer. Each token builds a query **Q**, key **K** and value **V**; attention is "
                    "`softmax(QKᵀ/√d)·V`, so words such as *personal data* or *farmers* get more weight when the model writes a risk. "
                    "Sentence-BERT maps text to vectors in one latent space; **cosine similarity** of a risk vector to each NIST guidance vector "
                    "(shown in the register tab) tells us which guidance supports it.")

if clean:
    key = (clean, mode, compare)
    if st.session_state.get("key") != key:
        with st.spinner("Drafting risks..."):
            runs = {}
            for m in (["zero-shot", "few-shot"] if compare else [mode]):
                rows, meta = core.draft_risks(clean, retr, m)
                df = core.build_register(rows, retr)
                runs[m] = (df, meta, core.grounding_metrics(df, retr, meta))
        st.session_state.update(key=key, runs=runs)
    runs = st.session_state["runs"]; df, meta, met = runs[mode if mode in runs else next(iter(runs))]

    with t2:
        st.markdown(f"**U2 · Drafted by:** `{meta['source']}` ({meta['mode']}) · **U3 · Grounded with:** {meta['retrieval']}")
        if meta["error"] and live: st.info(meta["error"] + " — offline template used.")
        for fn in core.FUNCTIONS:
            sub = df[df["RMF Function"] == fn]
            st.subheader(f"{fn}  ({len(sub)})")
            if len(sub):
                st.dataframe(sub.drop(columns=["RMF Function"]), width="stretch", hide_index=True)
            else:
                st.caption("No risk mapped to this function for this description.")
        st.markdown("**Latent-space view:** similarity of each risk to its best NIST guidance")
        st.bar_chart(df.set_index("ID")["Similarity"])
    with t3:
        st.pyplot(core.heatmap(df))
        st.download_button("⬇️ Download risk register (CSV)", core.to_csv(df), "risk_register.csv", "text/csv")
    with t4:
        st.markdown("**U5 · Measured hallucination / grounding test**")
        rows = [{"Prompt": m, **r[2], "source": r[1]["source"]} for m, r in runs.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("`grounding_rate` = share of drafted risks whose best NIST guidance match passes the similarity threshold. "
                    "Low values flag risks the LLM may have invented.")
        st.markdown("**Guardrails:** prompt-injection blocking, 3,000-character cap, and redaction of email, phone, Aadhaar and PAN before any text is sent to the LLM.  \n"
                    "**Licence / IP:** NIST AI RMF is a US-government publication; guidance here is paraphrased. Sentence-BERT MiniLM is Apache-2.0. LLM output is a draft for human review, not legal or compliance advice.")
else:
    for t in (t2, t3, t4): t.info("Enter a project description (20+ characters) to begin.")
