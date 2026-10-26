import os, sys, json, re, time
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = os.environ["OPENROUTER_MODEL"]
ORG = os.getenv("COMPANY_CONTEXT", "a generic company")

# ---------- Helpers ----------
def load_pages(path):
    if path.lower().endswith(".pdf"):
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages)]
    return [(1, open(path, encoding="utf-8").read())]

def chunk_pages(pages, size=3000):
    for page, text in pages:
        for i in range(0, len(text), size):
            part = text[i:i + size].strip()
            if part:
                yield page, part

def llm(system, user, retries=4):
    for n in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.1,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content
        except Exception as e:
            print("retry:", e)
            time.sleep(5 * (n + 1))
    raise RuntimeError("LLM failed after retries")

def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}

def norm(s):
    return re.sub(r"\s+", " ", s).lower().strip()

# ---------- Step 1: extract evidence ----------
EXTRACT_SYS = f"""You are a news/sentiment analyst for this organization: {ORG}
Read the text. Return ONLY JSON:
{{"items":[{{"quote":"<EXACT sentence copied from text>",
"sentiment":"positive|negative|neutral",
"reason":"<why it matters for the organization>"}}]}}
Only include items relevant to the organization. Never paraphrase quotes."""

def extract(pages):
    page_map = dict(pages)
    found = []
    for page, text in chunk_pages(pages):
        data = parse_json(llm(EXTRACT_SYS, text))
        page_text = norm(page_map[page])
        for it in data.get("items", []):
            it["page"] = page
            it["verified"] = norm(it.get("quote", ""))[:80] in page_text
            found.append(it)
    return found

# ---------- Step 2: verdict + actions ----------
SYNTH_SYS = f"""You advise leadership of: {ORG}
Given evidence items (quote, page, sentiment, reason), return ONLY JSON:
{{"verdict":"POSITIVE|NEGATIVE|MIXED","summary":"<3 sentences>",
"actions":[{{"action":"","priority":"High|Medium|Low","owner":"","why":""}}]}}
Verdict is about impact on the organization. Actions must be concrete."""

def synthesize(items):
    ev = [i for i in items if i["verified"] and i["sentiment"] != "neutral"]
    return ev, parse_json(llm(SYNTH_SYS, json.dumps(ev, indent=1)))

# ---------- Step 3: report ----------
def report(ev, res, out="output/report.md"):
    L = ["# Document Sentiment Report\n",
         f"**Verdict for us:** {res.get('verdict')}\n",
         res.get("summary", ""), "\n## Evidence (with references)"]
    for s in ("positive", "negative"):
        L.append(f"\n### {s.title()} news")
        for i in ev:
            if i["sentiment"] == s:
                L.append(f"- \"{i['quote']}\" (page {i['page']}) - {i['reason']}")
    L.append("\n## Recommended actions")
    for a in res.get("actions", []):
        L.append(f"- **[{a.get('priority')}]** {a.get('action')} "
                 f"(owner: {a.get('owner')}) - {a.get('why')}")
    os.makedirs("output", exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    evidence, result = synthesize(extract(load_pages(path)))
    report(evidence, result)
