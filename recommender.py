
import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY") or st.secrets.get("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Set it in .env or Streamlit secrets.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")


# =========================
# LOAD PICKLES (cached, runs once)
# =========================
@st.cache_resource
def load_resources():
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")

    title_to_idx = build_title_to_idx_map(indices_obj)
    return df, tfidf_matrix, tfidf_obj, title_to_idx


def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    title_to_idx: Dict[str, int] = {}
    for k, v in indices.items():
        title_to_idx[_norm_title(k)] = int(v)
    return title_to_idx


# =========================
# TMDB HELPERS (sync)
# =========================
def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY
    try:
        r = requests.get(f"{TMDB_BASE}{path}", params=q, timeout=20)
    except requests.RequestException as e:
        raise RuntimeError(f"TMDB request error: {type(e).__name__} | {e}")

    if r.status_code != 200:
        raise RuntimeError(f"TMDB error {r.status_code}: {r.text}")

    return r.json()


def tmdb_cards_from_results(results: List[dict], limit: int = 20) -> List[Dict]:
    out = []
    for m in (results or [])[:limit]:
        out.append({
            "tmdb_id": int(m["id"]),
            "title": m.get("title") or m.get("name") or "",
            "poster_url": make_img_url(m.get("poster_path")),
            "release_date": m.get("release_date"),
            "vote_average": m.get("vote_average"),
        })
    return out


def tmdb_movie_details(movie_id: int) -> Dict:
    data = tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return {
        "tmdb_id": int(data["id"]),
        "title": data.get("title") or "",
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "poster_url": make_img_url(data.get("poster_path")),
        "backdrop_url": make_img_url(data.get("backdrop_path")),
        "genres": data.get("genres", []) or [],
    }


def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return tmdb_get("/search/movie", {
        "query": query,
        "include_adult": "false",
        "language": "en-US",
        "page": page,
    })


def tmdb_search_first(query: str) -> Optional[dict]:
    data = tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def attach_tmdb_card_by_title(title: str) -> Optional[Dict]:
    try:
        m = tmdb_search_first(title)
        if not m:
            return None
        return {
            "tmdb_id": int(m["id"]),
            "title": m.get("title") or title,
            "poster_url": make_img_url(m.get("poster_path")),
            "release_date": m.get("release_date"),
            "vote_average": m.get("vote_average"),
        }
    except Exception:
        return None


# =========================
# TF-IDF
# =========================
def get_local_idx_by_title(title: str, title_to_idx: Dict[str, int]) -> int:
    key = _norm_title(title)
    if key in title_to_idx:
        return int(title_to_idx[key])
    raise ValueError(f"Title not found in local dataset: '{title}'")


def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    df, tfidf_matrix, _, title_to_idx = load_resources()

    idx = get_local_idx_by_title(query_title, title_to_idx)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)

    out = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out


# =========================
# PUBLIC FUNCTIONS (what Streamlit calls)
# =========================
def get_home_feed(category: str = "popular", limit: int = 24) -> List[Dict]:
    if category == "trending":
        data = tmdb_get("/trending/movie/day", {"language": "en-US"})
        return tmdb_cards_from_results(data.get("results", []), limit=limit)

    if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
        raise ValueError("Invalid category")

    data = tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    return tmdb_cards_from_results(data.get("results", []), limit=limit)


def get_movie_details(tmdb_id: int) -> Dict:
    return tmdb_movie_details(tmdb_id)


def get_genre_recommendations(tmdb_id: int, limit: int = 18) -> List[Dict]:
    details = tmdb_movie_details(tmdb_id)
    if not details["genres"]:
        return []
    genre_id = details["genres"][0]["id"]
    discover = tmdb_get("/discover/movie", {
        "with_genres": genre_id,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": 1,
    })
    cards = tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c["tmdb_id"] != tmdb_id]


def get_tfidf_recommendations(title: str, top_n: int = 10) -> List[Dict]:
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


def search_bundle(query: str, tfidf_top_n: int = 12, genre_limit: int = 12) -> Dict:
    best = tmdb_search_first(query)
    if not best:
        raise ValueError(f"No TMDB movie found for query: {query}")

    tmdb_id = int(best["id"])
    details = tmdb_movie_details(tmdb_id)

    try:
        recs = tfidf_recommend_titles(details["title"], top_n=tfidf_top_n)
    except Exception:
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    tfidf_items = []
    for title, score in recs:
        card = attach_tmdb_card_by_title(title)
        tfidf_items.append({"title": title, "score": score, "tmdb": card})

    genre_recs = []
    if details["genres"]:
        genre_id = details["genres"][0]["id"]
        discover = tmdb_get("/discover/movie", {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        })
        cards = tmdb_cards_from_results(discover.get("results", []), limit=genre_limit)
        genre_recs = [c for c in cards if c["tmdb_id"] != details["tmdb_id"]]

    return {
        "query": query,
        "movie_details": details,
        "tfidf_recommendations": tfidf_items,
        "genre_recommendations": genre_recs,
    }