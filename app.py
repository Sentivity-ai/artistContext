import os
import re
import hashlib
import asyncio
import aiohttp
import praw
import pandas as pd
import numpy as np

from flask import Flask, request, jsonify
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urljoin
from bs4 import BeautifulSoup
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# ─── Clients ────────────────────────────────────────────────────────────────

reddit = praw.Reddit(
    client_id=os.environ["REDDIT_CLIENT_ID"],
    client_secret=os.environ["REDDIT_CLIENT_SECRET"],
    user_agent="sentivityb2c",
    check_for_async=False,
)

analyzer = SentimentIntensityAnalyzer()

# ─── Helpers ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_text(x):
    return clean_text(x if x is not None else "")

def try_parse_datetime(value):
    if not value:
        return None
    value = value.strip()
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%b %d, %Y at %I:%M %p",
        "%b %d, %Y",
        "%B %d, %Y",
    ]:
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    return None

def make_row(*, source_name, title, body, url, dt, score=1, num_comments=0, comments_text=""):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    uid = hashlib.md5(f"{source_name}|{url}|{title}".encode("utf-8")).hexdigest()
    return {
        "id": uid,
        "title": safe_text(title),
        "selftext": safe_text(body),
        "score": score,
        "num_comments": num_comments,
        "subreddit": source_name,
        "created_utc": dt.timestamp(),
        "created_dt": dt,
        "url": url,
        "permalink": url,
        "comments_text": safe_text(comments_text),
    }

def build_df_from_rows(rows):
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "title", "selftext", "score", "num_comments", "subreddit",
            "created_utc", "created_dt", "url", "permalink", "comments_text", "full_text"
        ])
    df["full_text"] = (
        df["title"].fillna("") + "\n" +
        df["selftext"].fillna("") + "\n" +
        df["comments_text"].fillna("")
    ).str.strip()
    return df


def get_similar_subreddit(artist_name: str) -> Optional[str]:
    """Finds the most relevant subreddit for an artist name."""
    try:
        # We grab the top 1 result from Reddit's subreddit search
        search_results = list(reddit.subreddits.search(artist_name, limit=1))
        if search_results:
            return search_results[0].display_name
    except Exception as e:
        print(f"Subreddit similarity search failed: {e}")
    return None

# ─── Data Collection ─────────────────────────────────────────────────────────

def search_reddit(query: str, *, limit=None, deadline: float = None) -> pd.DataFrame:
    import time
    rows = []
    
    # We switch back to sort="hot" as requested, but keep the month filter
    # "Hot" is more reliable for general discovery than "Relevance"
    try:
        search_iterator = reddit.subreddit("all").search(
            query, sort="hot", time_filter="month", limit=limit
        )
        
        for post in search_iterator:
            if deadline and time.time() > deadline:
                break
                
            rows.append({
                "id": post.id,
                "title": clean_text(post.title),
                "selftext": clean_text(post.selftext),
                "score": post.score,
                "num_comments": post.num_comments,
                "subreddit": str(post.subreddit),
                "created_utc": post.created_utc,
                "created_dt": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                "url": post.url,
                "permalink": f"https://www.reddit.com{post.permalink}",
                "comments_text": "",
            })
    except Exception as e:
        print(f"Search failed for query [{query}]: {e}")
        
    df = pd.DataFrame(rows)
    if not df.empty:
        # Use fillna to prevent the entire full_text from becoming NaN if body is empty
        df["full_text"] = (
            df["title"].fillna("") + "\n" + 
            df["selftext"].fillna("") + "\n" + 
            df["comments_text"].fillna("")
        ).str.strip()
    return df

async def search_reddit_async(artist_name, limit=None, timeout=10.0):
    import time
    deadline = time.time() + timeout
    
    # We removed the complex boolean logic and quotes to make the search "fuzzier"
    # This ensures Reddit actually finds matches even if phrasing is slightly off
    contexts = [
        "Tours", "Fan club", "Songs", "Sold out", "Albums", 
        "Setlist", "Merch", "Festival", "Meet and greet", 
        "Encore", "Pre-save", "Outfit planning", "Radio"
    ]
    
    # Create tasks: search for [Artist Name Context]
    tasks = [
        asyncio.to_thread(
            search_reddit, f"{artist_name} {ctx}", limit=limit, deadline=deadline
        )
        for ctx in contexts
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Filter out empty results before merging
    valid_dfs = [df for df in results if not df.empty]
    
    if not valid_dfs:
        return pd.DataFrame()
    
    # Combine everything and drop duplicates (essential for multi-query)
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["id"])
    
    return combined_df



POPJUSTICE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://forum.popjustice.com/",
}

async def scrape_popjustice(session, artist_name, max_results=None):
    source = "Popjustice"
    q = quote_plus(artist_name)
    search_url = (
        f"https://forum.popjustice.com/search/search"
        f"?keywords={q}&c[title_only]=0&o=date"
    )
    rows = []
    try:
        async with session.get(search_url, headers=POPJUSTICE_HEADERS, timeout=aiohttp.ClientTimeout(total=25)) as resp:
            html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        candidates = soup.select('a[href*="/threads/"]')
        seen = set()
        for a in candidates:
            href = a.get("href")
            title = safe_text(a.get_text(" ", strip=True))
            if not href or not title:
                continue
            full_url = urljoin("https://forum.popjustice.com", href)
            if full_url in seen:
                continue
            seen.add(full_url)
            parent = a.find_parent(["div", "li", "article"])
            parent_text = safe_text(parent.get_text(" ", strip=True) if parent else "")
            dt = None
            time_tag = parent.find("time") if parent else None
            if time_tag:
                dt = (
                    try_parse_datetime(time_tag.get("datetime")) or
                    try_parse_datetime(time_tag.get_text(" ", strip=True))
                )
            row = make_row(source_name=source, title=title, body=parent_text, url=full_url, dt=dt)
            if row is not None:
                rows.append(row)
            if len(rows) >= max_results:
                break
    except Exception as e:
        print(f"Popjustice scrape failed: {repr(e)}")
    return build_df_from_rows(rows)

async def collect_mentions_async(artist_name, reddit_query, limit=None):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            search_reddit_async(reddit_query, limit=limit),
            scrape_popjustice(session, artist_name),
            return_exceptions=True,
        )
    source_names = ["Reddit", "Popjustice"]
    cleaned = {}
    for name, result in zip(source_names, results):
        if isinstance(result, Exception):
            print(f"{name} failed: {result}")
            cleaned[name] = pd.DataFrame()
        else:
            cleaned[name] = result.copy()
    return cleaned

# ─── Analytics ───────────────────────────────────────────────────────────────

def compute_weekly_mentions(combined_df, artist_name) -> tuple[int, list[float]]:
    """Returns (total_mentions_35d, [week1_avg, week2_avg, week3_avg, week4_avg, week5_avg])"""
    combined_df = combined_df.copy()
    combined_df["created_dt"] = pd.to_datetime(combined_df["created_dt"], utc=True, errors="coerce")
    combined_df = combined_df.dropna(subset=["created_dt"])
    combined_df["date"] = combined_df["created_dt"].dt.normalize()

    latest_date = combined_df["date"].max()
    last_35_start = latest_date - pd.Timedelta(days=34)

    last_35_df = combined_df[
        (combined_df["date"] >= last_35_start) &
        (combined_df["date"] <= latest_date)
    ].copy()

    total_mentions = len(last_35_df)

    last_35_df["days_from_start"] = (last_35_df["date"] - last_35_start).dt.days
    last_35_df["week_number"] = (last_35_df["days_from_start"] // 7) + 1

    weekly = (
        last_35_df.groupby("week_number")
        .size()
        .reset_index(name="mentions")
    )

    all_weeks = pd.DataFrame({"week_number": [1, 2, 3, 4, 5]})
    weekly = all_weeks.merge(weekly, on="week_number", how="left").fillna(0)
    weekly["mentions"] = weekly["mentions"].astype(float)

    return total_mentions, weekly["mentions"].tolist()

# ─── Route ───────────────────────────────────────────────────────────────────

@app.route("/map", methods=["POST"])
def map_artist():
    data = request.get_json(force=True)
    artist = (data.get("artist") or "").strip()
    context = (data.get("context") or "").strip()

    if not artist:
        return jsonify({"error": "Missing required field: artist"}), 400

    reddit_query = f"{artist} {context}".strip()

    try:
        source_dfs = asyncio.run(collect_mentions_async(artist, reddit_query, limit=None))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        source_dfs = loop.run_until_complete(collect_mentions_async(artist, reddit_query, limit=None))
        loop.close()

    all_dfs = []
    for source_name, df in source_dfs.items():
        if df is not None and not df.empty:
            df = df.copy()
            df["company"] = artist
            df["source"] = source_name
            if "full_text" not in df.columns:
                df["full_text"] = (
                    df["title"].fillna("") + "\n" +
                    df["selftext"].fillna("") + "\n" +
                    df["comments_text"].fillna("")
                ).str.strip()
            all_dfs.append(df)

    if not all_dfs:
        return jsonify({
            "artist": artist,
            "mention_count": 0,
            "weekly_mentions": [0.0, 0.0, 0.0, 0.0, 0.0],
        })

    combined_df = pd.concat(all_dfs, ignore_index=True)
    mention_count, weekly_mentions = compute_weekly_mentions(combined_df, artist)

    return jsonify({
        "artist": artist,
        "mention_count": mention_count,
        "weekly_mentions": weekly_mentions,
    })


@app.route("/map/<artist_name>", methods=["GET"])
@app.route("/map/<artist_name>/<context>", methods=["GET"])
def map_artist_get(artist_name, context="songs"):
    from flask import request as flask_request
    # Reuse the same logic by faking a POST body
    flask_request.environ["REQUEST_METHOD"] = "GET"
    artist = artist_name.strip()
    reddit_query = f"{artist} {context}".strip()

    try:
        source_dfs = asyncio.run(collect_mentions_async(artist, reddit_query, limit=None))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        source_dfs = loop.run_until_complete(collect_mentions_async(artist, reddit_query, limit=None))
        loop.close()

    all_dfs = []
    for source_name, df in source_dfs.items():
        if df is not None and not df.empty:
            df = df.copy()
            df["company"] = artist
            df["source"] = source_name
            if "full_text" not in df.columns:
                df["full_text"] = (
                    df["title"].fillna("") + "\n" +
                    df["selftext"].fillna("") + "\n" +
                    df["comments_text"].fillna("")
                ).str.strip()
            all_dfs.append(df)

    if not all_dfs:
        return jsonify({
            "artist": artist,
            "mention_count": 0,
            "weekly_mentions": [0.0, 0.0, 0.0, 0.0, 0.0],
        })

    combined_df = pd.concat(all_dfs, ignore_index=True)
    mention_count, weekly_mentions = compute_weekly_mentions(combined_df, artist)

    return jsonify({
        "artist": artist,
        "mention_count": mention_count,
        "weekly_mentions": weekly_mentions,
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
