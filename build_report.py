"""Builds RollNo_Name_P55.pdf from results/. Re-run after eval.py with your real LLM key + deployed URLs."""
from xml.sax.saxutils import escape
import glob, inspect, json, os, textwrap
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, Preformatted, KeepTogether)
import core

APP_URL = os.getenv("APP_URL", "<add live app URL after deployment>")
GITHUB = os.getenv("GITHUB_URL", "<add GitHub repo URL>")
OUT = os.getenv("REPORT_NAME", "55_Vinit_Kale_P55.pdf")
M = json.load(open("results/metrics.json"))
S = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=S["Heading1"], fontSize=15, spaceBefore=10, textColor=colors.HexColor("#1f3a5f"))
B = ParagraphStyle("B", parent=S["BodyText"], fontSize=10, leading=14)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#444444"), alignment=1)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10)
def P(t, s=B): return Paragraph(t, s)

def block_diagram(path):
    fig, ax = plt.subplots(figsize=(9, 3.2)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 36)
    boxes = [(1, 22, "Project\ndescription"), (18, 22, "Guardrail\n(injection, PII)"), (36, 22, "Sentence-BERT\nretrieves RMF\nguidance"),
             (56, 22, "LLM drafts risks\n(zero / few-shot)"), (77, 22, "Validate JSON\n+ schema")]
    boxes2 = [(77, 4, "Map to Govern /\nMap / Measure /\nManage (SBERT)"), (56, 4, "Score L x I\n+ grounding test"), (36, 4, "Heatmap image\n+ CSV export"), (14, 4, "Streamlit UI\n(live app)")]
    for x, y, t in boxes + boxes2:
        ax.add_patch(FancyBboxPatch((x, y), 17, 10, boxstyle="round,pad=0.3", fc="#e8f0fb", ec="#1f3a5f"))
        ax.text(x + 8.5, y + 5, t, ha="center", va="center", fontsize=7.5)
    for i in range(4): ax.annotate("", (boxes[i + 1][0] - 0.3, 27), (boxes[i][0] + 17.3, 27), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", (85.5, 14.5), (85.5, 21.7), arrowprops=dict(arrowstyle="->"))
    for i in range(3): ax.annotate("", (boxes2[i + 1][0] + 17.3, 9), (boxes2[i][0] - 0.3, 9), arrowprops=dict(arrowstyle="->"))
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)

def tbl(data, widths, fs=7.5, head=True):
    t = Table([[Paragraph(escape(str(c)), ParagraphStyle("c", parent=SM, fontSize=fs, leading=fs + 2)) for c in r] for r in data], colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe6f5")))
    t.setStyle(TableStyle(st)); return t

def code(fn):
    src = inspect.getsource(fn)
    lines = [l2 for l in src.splitlines() for l2 in (textwrap.wrap(l, 100, subsequent_indent="      ", replace_whitespace=False) or [""])]
    return Preformatted("\n".join(lines), ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8, backColor=colors.HexColor("#f4f4f4")))

block_diagram("results/block_diagram.png")
runs = pd.DataFrame(M["runs"]); regs = {r["project"]: pd.read_csv(f"results/{r['project']}_register.csv") for r in M["runs"] if r["mode"] == "few-shot"}
offline = (not M["llm_configured"]) or any(r["source"] != "llm" for r in M["runs"])
story = []
# 1 cover
story += [Spacer(1, 3 * cm), P("Generative AI - CA Mini-Project", ParagraphStyle("t0", parent=B, fontSize=12, alignment=1)),
          P("P55 - NIST AI RMF Risk Register Builder", ParagraphStyle("t", parent=S["Title"], fontSize=22, leading=26)), Spacer(1, 1 * cm),
          tbl([["Name", "Vinit Kale"], ["Roll No.", "55"], ["Class", "LY AIA 3"], ["Problem", "P55 (Track E - Ethics, governance and responsible deployment)"],
               ["Live app URL", APP_URL], ["GitHub repo", GITHUB], ["Demo video", "<add video link>"]], [4 * cm, 11 * cm], fs=10, head=False), PageBreak()]
# 2 problem
story += [P("1. Problem and objective", H1),
          P("Teams building AI systems often skip structured risk analysis because it is slow and unfamiliar. The goal of this project is to turn a plain-language AI project description into a first-draft "
            "<b>risk register</b> aligned to the NIST AI Risk Management Framework (AI RMF 1.0), whose four functions are <b>Govern, Map, Measure and Manage</b>. "
            "The app drafts risks with an LLM, grounds each risk in retrieved NIST guidance using Sentence-BERT, scores likelihood x impact, draws a risk heatmap and exports a CSV."),
          P("Mapping of the five required tasks: <b>U1</b> lifecycle stages to risk points; <b>U2</b> LLM drafts risks; <b>U3</b> Sentence-BERT retrieves guidance; <b>U4</b> heatmap image and CSV; <b>U5</b> risks organised under the four RMF functions with a measured grounding test and guardrails.")]
# 3 tools
story += [P("2. Tools, models and dataset", H1), tbl([["Item", "Choice"],
    ["Language / UI / hosting", "Python 3, Streamlit, Streamlit Community Cloud (or Hugging Face Spaces)"],
    ["GPT-style LLM", "Open model hosted by Groq API (default llama-3.1-8b-instant, configurable); Transformer decoder; key kept in platform secrets"],
    ["BERT-family model", "sentence-transformers/all-MiniLM-L6-v2 (Sentence-BERT, Apache-2.0), PyTorch backend"],
    ["Image modality", "Matplotlib risk heatmap generated for every register"],
    ["Knowledge base", "36 short paraphrases: 19 AI RMF categories (GOVERN 1-6, MAP 1-5, MEASURE 1-4, MANAGE 1-4), 7 trustworthiness characteristics, 10 generative-AI risks (NIST AI 600-1)"],
    ["Fallbacks", "TF-IDF retrieval if Sentence-BERT cannot load; rule-based drafter if no LLM key or invalid JSON"],
    ["Fine-tuning", "Not performed; prompt-based learning (zero-shot vs few-shot) is used instead and compared"]], [4.2 * cm, 12.3 * cm], fs=8)]
# 4 approach
story += [P("3. Approach and block diagram", H1), Image("results/block_diagram.png", width=16.5 * cm, height=5.9 * cm), P("Figure 1. System block diagram.", CAP),
          P("<b>Generative vs discriminative:</b> the LLM is generative (it writes new risk statements), while Sentence-BERT is used discriminatively (it scores which guidance text is closest). "
            "<b>Attention:</b> the LLM computes softmax(QK<super>T</super>/sqrt(d))V, so words such as 'personal data' receive more weight when a privacy risk is written. "
            "<b>Latent-space view:</b> risk and guidance sentences are embedded in one vector space and compared by cosine similarity; the similarity bar chart in the app shows this. "
            "<b>Prompt techniques:</b> zero-shot (schema only) and few-shot (schema plus one worked example) prompts are both implemented.")]
# 5 code
story += [P("4. Key code", H1), P("Full source: " + escape(GITHUB), SM), P("Input guardrail (sanitize) and Sentence-BERT retrieval (Retriever.search):", SM), code(core.sanitize), Spacer(1, 4), code(core.Retriever.search),
          Spacer(1, 4), P("Grounding (hallucination) test:", SM), code(core.grounding_metrics)]
# 6 screenshots
story += [PageBreak(), P("5. Output screenshots", H1)]
for n, (name, df) in enumerate(regs.items(), 1):
    story += [Image(f"results/{name}_heatmap.png", width=9.2 * cm, height=7.5 * cm), P(f"Figure {n + 1}. Risk heatmap generated for sample project {name}.", CAP)]
shots = sorted(glob.glob("screenshots/*.png"))
for k, p in enumerate(shots, 1):
    story += [Image(p, width=15 * cm, height=8.5 * cm, kind="proportional"), P(f"Figure {len(regs) + 1 + k}. Live app screenshot: {os.path.basename(p)}.", CAP)]
if not shots: story.append(P("<i>Live-app screenshots: add PNG files to the screenshots/ folder and re-run build_report.py.</i>", SM))
# 7 results
story += [PageBreak(), P("6. Results", H1),
          P(f"Run mode: <b>LLM {'connected' if M['llm_configured'] else 'not connected (offline template drafter)'}</b>; retriever: <b>{M['retrieval_backend']}</b>."
            + (" Because no LLM key and no Sentence-BERT download were available when this table was generated, the numbers below test the pipeline, mapping, scoring and guardrails, but <b>do not measure LLM quality</b>; re-run eval.py on the deployed setup to compare zero-shot and few-shot." if offline else ""))]
rt = [["Project", "Prompt", "Drafted by", "Risks", "Grounding rate", "Schema valid", "Functions covered"]] + \
     [[r.project, r["mode"], r.source, r.risks, r.grounding_rate, r.schema_valid_rate, f"{r.functions_covered}/4"] for _, r in runs.iterrows()]
story += [tbl(rt, [4.2 * cm, 1.8 * cm, 2.6 * cm, 1.2 * cm, 2.2 * cm, 2.1 * cm, 2.4 * cm]), P("Table 1. Grounding (hallucination) test and structure checks.", CAP), Spacer(1, 6)]
gt = [["Guardrail test input", "Expected", "Got", "Pass"]] + [[g["input"][:70], g["expected"], g["got"], "PASS" if g["pass"] else "FAIL"] for g in M["guardrail_tests"]]
story += [tbl(gt, [8.5 * cm, 2.5 * cm, 2.5 * cm, 1.5 * cm]), P("Table 2. Guardrail pass/fail.", CAP), Spacer(1, 6)]
for name, df in regs.items():
    view = [["ID", "Function", "Ref", "Stage", "Risk", "L", "I", "Rating", "Mitigation"]] + \
           [[r["ID"], r["RMF Function"], r["RMF Ref"], r["Stage"], r["Risk"], r["Likelihood"], r["Impact"], r["Rating"], r["Mitigation"]] for _, r in df.iterrows()]
    story += [KeepTogether([P(f"<b>Risk register - {name}</b>", B), tbl(view, [0.9 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 4.0 * cm, 0.5 * cm, 0.5 * cm, 1.3 * cm, 3.8 * cm], fs=6.3),
                            P(f"Table. Register for {name} (full CSV in results/).", CAP)]), Spacer(1, 6)]
# 8 limits
weak = {n: d.loc[d["Similarity"].idxmin()] for n, d in regs.items()}
story += [P("7. Limitations and failure cases", H1)]
for n, r in weak.items():
    story.append(P(f"<b>Failure case ({n}):</b> risk {r['ID']} ('{r['Risk'][:110]}...') had the weakest guidance match (similarity {r['Similarity']}) and was mapped to {r['RMF Ref']}. "
                   "The match is only loosely related, so the function label for this row should be reviewed by a person.", B))
story += [P("Other limitations: likelihood and impact are opinions not measurements; function mapping uses embedding similarity and can mislabel governance-style risks as MAP; "
            "only two sample projects were tested; the knowledge base is a small paraphrased subset of NIST text; PII redaction is regex-based; no fine-tuning was done.")]
# 9 ethics
story += [P("8. Ethical note", H1), P("<b>Privacy:</b> project descriptions may contain personal data, so emails, phone numbers, Aadhaar and PAN patterns are redacted before text reaches the LLM API; nothing is stored. "
          "<b>Bias:</b> LLM-drafted risks may overlook harms to under-represented groups and languages; a human reviewer must check every row. "
          "<b>IP / licence:</b> the NIST AI RMF is a US-government publication and is paraphrased here; MiniLM is Apache-2.0; LLM use follows the provider's terms. "
          "<b>Misuse:</b> the register is a brainstorming aid, not legal or compliance advice, and prompt injection is mitigated by filtering but not eliminated.")]
# 10 conclusion
story += [P("9. Conclusion", H1), P("The builder converts a short AI project description into a grounded, scored risk register organised under Govern, Map, Measure and Manage, with a heatmap and CSV export, and covers all five syllabus units in one deployable app. "
          "The grounding test and guardrails make its weaknesses visible instead of hidden. Future work: evaluate more projects, compare zero-shot and few-shot on a live LLM, and add a human-review column.")]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.drawString(2 * cm, 1 * cm, "55_Vinit Kale - P55 NIST AI RMF Risk Register Builder"); c.drawRightString(19 * cm, 1 * cm, f"Page {d.page}"); c.restoreState()
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="P55 NIST AI RMF Risk Register Builder", author="Vinit Kale").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
