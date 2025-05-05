import os, sys, json, time, requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import praw
import yfinance as yf
from serpapi import GoogleSearch
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY  = os.getenv("SERPAPI_API_KEY")

REDDIT = praw.Reddit(
    client_id     = os.getenv("REDDIT_CLIENT_ID"),
    client_secret = os.getenv("REDDIT_CLIENT_SECRET"),
    username      = os.getenv("REDDIT_USERNAME"),
    password      = os.getenv("REDDIT_PASSWORD"),
    user_agent    = os.getenv("REDDIT_UA") or "script:invest:v1.0 (by u/anonymous)",
)

GPT_MODEL = "gpt-4.1-nano"
SYSTEM_PROMPT = (
    "You are a helpful, concise financial assistant. "
    "Base every answer strictly on the supplied text; never invent facts."
    "Keep all responses proffesional, impersonal, and analytical"
    "You should be evaluating facts solely on their financial impact. Yoy may consider social, economic, and other impacts, but at the end of the day consider only how they will impact markets or specific comoddities."
)

# GPT pricing (USD per million tokens)
INPUT_COST  = 0.10 / 1_000_000
OUTPUT_COST = 0.40 / 1_000_000
TOKENS_USED = {"in": 0, "out": 0}

# counts & limits
COUNTS = dict(market=10, company=10, reddit_posts=25)
TOKENS = dict(blurb=200, synth=160, reddit_token=180)
DEFAULT_SUBREDDITS = ["stocks", "investing", "wallstreetbets", "options"]

GENERIC_TOKENS = {
    "stock", "stocks", "share", "shares", "market", "markets",
    "invest", "investing", "trade", "trading", "call", "puts",
}

PAYWALL_TAGS  = [".paywall",".subscription-wall","#gateway-content",".meteredContent"]
PAYWALL_CACHE = "domain_status.json"
NAME_CACHE    = "ticker_names.json"

PROMPTS = {
    # news
    "market_blurb"  : "Summarise the following news article in 2 to 3 sentences. Keep information relevant to the now and the most recent and urgemt events. Stay fully neutral in reporting news. The news should be analyzed through a financial lens, how markets will be impacted, how investor sentiment changes, etc.",
    "company_blurb" : "Summarise the following news article in 2 to 3 sentences, focusing on company‑specific impacts. Your lens should be as a person looking to invest. Consider recenet events in context of their past decisions and the overall economy. Be impartial, be just as likely to conclude a company will increase in value as it is to decrease",
    "market_synth"  : "Using ONLY the bullet points below, write a 2–3 sentence overview of the main themes for this industry. Keep them relevant to the now, not old trends. It is relevant to mention recent changes and directions.",
    "company_synth" : "Using ONLY the bullet points below, write a 2–3 sentence overview of the main company‑specific themes. Focus on financial impact.",
    # reddit
    "reddit_terms"  : (
        "Suggest up to five short, single‑word Reddit search tokens that retail "
        "investors would likely use when discussing COMPANY (INDUSTRY industry). "
        "Return them comma‑separated, no other text."
    ),
    "reddit_token_synth": (
        "Given ONLY the Reddit post bullets below, write a concise 2‑sentence "
        "summary including prevailing sentiment (positive / neutral / negative)."
    ),
    "reddit_final_synth": (
        "Using ONLY the token‑level summaries, provide a short overall Reddit "
        "sentiment (positive / neutral / negative) and key investor themes."
    ),
}

client = OpenAI(api_key=OPENAI_API_KEY)

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to install dependencies.")
        print(e)

def gpt(prompt: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt}
        ],
        temperature=0.25,
        max_tokens=max_tokens,
    )
    TOKENS_USED["in"]  += resp.usage.prompt_tokens
    TOKENS_USED["out"] += resp.usage.completion_tokens
    return resp.choices[0].message.content.strip()

def load_name_cache() -> dict:
    try: return json.load(open(NAME_CACHE))
    except Exception: return {}

def save_name_cache(d: dict) -> None:
    json.dump(d, open(NAME_CACHE,"w"), indent=2)

def company_name(ticker: str) -> str:
    cache = load_name_cache()
    if ticker in cache:
        return cache[ticker]

    log(f"   ↳ fetching company name for {ticker} from yfinance …")
    try:
        name = yf.Ticker(ticker).info.get("shortName") or ticker
    except Exception:
        name = ticker

    cache[ticker] = name
    save_name_cache(cache)
    return name

def google_news(query: str, n: int, timeframe: str = "w") -> list[dict]:
    """
    timeframe: 'h' = last hour, 'd' = 24 h, 'w' = 7 days (default),
               'm' = 30 days, 'y' = past year   — Google tbs=qdr: codes
    """
    params = {
        "engine": "google_news",
        "q": query,
        "tbs": f"qdr:{timeframe}",
        "num": n,
        "hl": "en",
        "gl": "us",
        "api_key": SERPAPI_API_KEY,
    }
    news = GoogleSearch(params).get_dict().get("news_results", [])
    return [r for r in news if r.get("link")][:n]

def is_accessible(url: str) -> bool:
    dom = ".".join(urlparse(url).netloc.split(".")[-2:])
    try: status = json.load(open(PAYWALL_CACHE))
    except Exception: status = {}
    if dom in status: return status[dom]

    try:
        html = requests.get(url, timeout=6).text
        ok = not any(BeautifulSoup(html,"html.parser").select_one(t) for t in PAYWALL_TAGS)
    except requests.RequestException:
        ok = False

    status[dom] = ok
    json.dump(status, open(PAYWALL_CACHE,"w"), indent=2)
    return ok

def article_text(url: str) -> str | None:
    try: html = requests.get(url, timeout=6).text
    except requests.RequestException: return None
    return " ".join(
        p.get_text(" ", strip=True)
        for p in BeautifulSoup(html,"html.parser").find_all("p")
    )

def industry_of(ticker: str) -> str:
    return gpt(f"For ticker '{ticker}', respond ONLY with its primary industry sector.", 12)

def make_blurb(text: str, key: str) -> str:
    return gpt(f"{PROMPTS[key]}\n\n{text}", TOKENS["blurb"])

def synthesize(bullets: list[str], key: str, max_tok: int) -> str:
    joined = "\n".join(f"- {b}" for b in bullets)
    return gpt(f"{PROMPTS[key]}\n{joined}", max_tok)

def news_summary(ticker: str,
                 scope: str,
                 num_results: int = 15,
                 timeframe: str = "w",
                 keyword: str | None = None):

    query = f"{ticker} {scope}"
    if keyword:
        query += f" {keyword}"

    params = {
        "engine": "google",
        "tbm": "nws",
        "q": query,
        "tbs": f"qdr:{timeframe}",
        "num": num_results,
        "api_key": SERPAPI_API_KEY,
    }
    raw_articles = GoogleSearch(params).get_dict().get("news_results", [])
    links = [a["link"] for a in raw_articles if a.get("link")]

    good_links = [u for u in links if is_accessible(u)]

    key = "market_blurb" if scope == "market" else "company_blurb"
    blurbs = []
    for url in good_links:
        txt = article_text(url) or ""
        if txt:
            blurbs.append(make_blurb(txt[:6_000], key))

    if not blurbs:
        return f"**{scope.title()} news for {ticker}**\nNo accessible articles."

    syn_key = "market_synth" if scope == "market" else "company_synth"
    overview = synthesize(blurbs, syn_key, TOKENS["synth"])

    markdown_summary = (
        f"### {scope.title()} news for **{ticker}**\n"
        f"{overview}\n\n"
        + "\n".join(f"- {b}" for b in blurbs)
    )
    return markdown_summary

def reddit_terms(company: str, industry: str) -> list[str]:
    log(f"[reddit] generating search tokens …")
    raw = gpt(
        PROMPTS["reddit_terms"].replace("COMPANY", company).replace("INDUSTRY", industry),
        40,
    )
    tokens = [
        t.strip() for t in raw.split(",")
        if t.strip() and t.lower() not in GENERIC_TOKENS and len(t) >= 2
    ]
    log(f"[reddit] tokens → {tokens}")
    return tokens

def fetch_reddit_posts(token: str, n: int, subs: list[str], comp_name: str) -> list[dict]:
    query = f"{comp_name} {token}"
    multi = "+".join(subs)
    log(f"[reddit] pulling {n} top posts for “{query}” …")
    subs_iter = REDDIT.subreddit(multi).search(
        query, sort="top", time_filter="week", limit=n
    )
    submissions = sorted(list(subs_iter), key=lambda s: s.score, reverse=True)[:n]

    results=[]
    for s in submissions:
        s.comments.replace_more(limit=0)
        top_comments = [c.body for c in s.comments[:3]]
        results.append({
            "title": s.title,
            "selftext": s.selftext[:200],
            "comments": " ".join(top_comments),
        })
    log(f"[reddit]   • {len(results)} posts retrieved")
    return results

def token_sentiment(posts: list[dict], token: str) -> str | None:
    if not posts:
        log(f"[reddit]   • no posts for “{token}” – skipping")
        return None
    t0 = time.perf_counter()
    bullets = [f"{p['title']} {p['selftext']} {p['comments']}" for p in posts]
    out = synthesize(bullets, "reddit_token_synth", TOKENS["reddit_token"])
    log(f"[reddit]   ✔ summarised “{token}” ({time.perf_counter()-t0:.1f}s)")
    return out

def reddit_summary(ticker: str) -> str:
    start = time.perf_counter()
    comp = company_name(ticker)
    ind  = industry_of(ticker)
    tokens = reddit_terms(comp, ind)

    token_blurbs=[]
    for tok in tokens:
        posts = fetch_reddit_posts(tok, COUNTS["reddit_posts"], DEFAULT_SUBREDDITS, comp)
        if (b := token_sentiment(posts, tok)):
            token_blurbs.append(b)

    if not token_blurbs:
        return f"**Reddit sentiment for {ticker}**\nNo substantial Reddit data."

    log(f"[reddit] synthesising overall sentiment …")
    overall = synthesize(token_blurbs, "reddit_final_synth", TOKENS["synth"])
    log(f"[reddit] ✔ reddit summary ready ({time.perf_counter()-start:.1f}s)")
    return f"**Reddit sentiment for {ticker}**\n{overall}"

if __name__ == "__main__":
    if "--install" in sys.argv:
        install_requirements()

    if len(sys.argv) < 2:
        sys.exit("Usage: python invest_agent.py <TICKER> [news|reddit|market|company|both]")

    ticker = sys.argv[1].upper()
    mode   = sys.argv[2].lower() if len(sys.argv) == 3 else "both"

    t0 = time.perf_counter()

    if mode in ("news","market","company","both"):
        if mode in ("news","market","both"):
            print(news_summary(ticker,"market"),"\n")
        if mode in ("news","company","both"):
            print(news_summary(ticker,"company"),"\n")

    if mode in ("reddit","both"):
        print(reddit_summary(ticker))

    cost = TOKENS_USED["in"]*INPUT_COST + TOKENS_USED["out"]*OUTPUT_COST
    log(f"[cost] prompt={TOKENS_USED['in']}  completion={TOKENS_USED['out']}  →  ${cost:.4f}")
    log(f"[done] total runtime {time.perf_counter()-t0:.1f}s")