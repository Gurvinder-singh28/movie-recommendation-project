import asyncio
import difflib
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"
TMDB_IMG_1280 = "https://image.tmdb.org/t/p/w1280"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# =========================
# GLOBALS
# =========================
df: Optional[pd.DataFrame] = None
tfidf_matrix: Any = None
TITLE_TO_IDX: Dict[str, int] = {}
ALL_TITLES: List[str] = []

http_client: Optional[httpx.AsyncClient] = None

MAX_RETRIES = 5
CACHE_TTL = 600  # seconds
_cache: Dict[Any, Tuple[float, Any]] = {}
_sem = asyncio.Semaphore(6)  # max parallel TMDB requests
_genre_map: Dict[int, str] = {}


def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def _load_pickle(name: str):
    with open(os.path.join(BASE_DIR, name), "rb") as f:
        return pickle.load(f)


# =========================
# STARTUP / SHUTDOWN
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, tfidf_matrix, TITLE_TO_IDX, ALL_TITLES, http_client

    # local dataset is optional: the app still works without it
    try:
        df = _load_pickle("df.pkl")
        indices = _load_pickle("indices.pkl")
        tfidf_matrix = _load_pickle("tfidf_matrix.pkl")
        TITLE_TO_IDX = {_norm_title(k): int(v) for k, v in indices.items()}
        ALL_TITLES = list(TITLE_TO_IDX.keys())
    except Exception as e:
        print(f"[WARN] local TF-IDF dataset not loaded: {e}")
        df, tfidf_matrix, TITLE_TO_IDX, ALL_TITLES = None, None, {}, []

    http_client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(retries=3),
        timeout=httpx.Timeout(20.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=10, max_keepalive_connections=5, keepalive_expiry=15
        ),
    )

    yield

    await http_client.aclose()


# =========================
# APP
# =========================
app = FastAPI(title="Movie Recommender API", version="5.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


# =========================
# MODELS
# =========================
class MovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class MovieDetails(BaseModel):
    tmdb_id: int
    title: str
    tagline: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    runtime: Optional[int] = None
    vote_average: Optional[float] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class Recommendations(BaseModel):
    tfidf: List[MovieCard]
    genre: List[MovieCard]


# =========================
# TMDB HELPERS
# =========================
def img(path: Optional[str], base: str = TMDB_IMG_500) -> Optional[str]:
    return f"{base}{path}" if path else None


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Cached TMDB GET with retries, backoff and stale-cache fallback."""
    key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL:
        return hit[1]

    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    last_error = "unknown error"
    for attempt in range(MAX_RETRIES):
        try:
            async with _sem:
                r = await http_client.get(f"{TMDB_BASE}{path}", params=q)

            if r.status_code == 200:
                data = r.json()
                if len(_cache) > 1000:
                    _cache.clear()
                _cache[key] = (time.time(), data)
                return data

            if r.status_code in (429, 500, 502, 503, 504):
                last_error = f"TMDB status {r.status_code}"
            elif r.status_code == 404:
                raise HTTPException(status_code=404, detail="Not found on TMDB")
            else:
                raise HTTPException(status_code=502, detail=f"TMDB error {r.status_code}")

        except httpx.RequestError as e:
            last_error = type(e).__name__

        await asyncio.sleep(0.5 * (2 ** attempt))

    if hit:
        return hit[1]

    raise HTTPException(status_code=502, detail=f"Could not reach TMDB ({last_error})")


async def safe_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Like tmdb_get but never raises: returns {} on any failure."""
    try:
        return await tmdb_get(path, params)
    except Exception:
        return {}


def to_cards(results: List[dict], limit: int = 24) -> List[MovieCard]:
    out: List[MovieCard] = []
    for m in (results or [])[:limit]:
        if not m.get("id"):
            continue
        out.append(
            MovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=img(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def movie_details(movie_id: int) -> MovieDetails:
    d = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return MovieDetails(
        tmdb_id=int(d["id"]),
        title=d.get("title") or "",
        tagline=d.get("tagline"),
        overview=d.get("overview"),
        release_date=d.get("release_date"),
        runtime=d.get("runtime"),
        vote_average=d.get("vote_average"),
        poster_url=img(d.get("poster_path")),
        backdrop_url=img(d.get("backdrop_path"), TMDB_IMG_1280),
        genres=d.get("genres", []) or [],
    )


async def search_first(title: str) -> Optional[dict]:
    data = await safe_get(
        "/search/movie",
        {"query": title, "include_adult": "false", "language": "en-US", "page": 1},
    )
    res = data.get("results", [])
    return res[0] if res else None


async def get_genre_map() -> Dict[int, str]:
    if not _genre_map:
        data = await safe_get("/genre/movie/list", {"language": "en-US"})
        for g in data.get("genres", []):
            _genre_map[int(g["id"])] = g["name"]
    return _genre_map


async def genre_cards(details: MovieDetails, limit: int) -> List[MovieCard]:
    if not details.genres:
        return []
    data = await safe_get(
        "/discover/movie",
        {
            "with_genres": details.genres[0]["id"],
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = to_cards(data.get("results", []), limit + 1)
    return [c for c in cards if c.tmdb_id != details.tmdb_id][:limit]


# =========================
# LOCAL DATASET (optional signal)
# =========================
def find_local_idx(title: str) -> Optional[int]:
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return TITLE_TO_IDX[key]
    close = difflib.get_close_matches(key, ALL_TITLES, n=1, cutoff=0.85)
    return TITLE_TO_IDX[close[0]] if close else None


def local_neighbour_titles(title: str, top_n: int = 10) -> List[str]:
    """Nearest titles from your own df.pkl / tfidf_matrix.pkl. Never raises."""
    try:
        if df is None or tfidf_matrix is None:
            return []
        idx = find_local_idx(title)
        if idx is None:
            return []
        scores = (tfidf_matrix @ tfidf_matrix[idx].T).toarray().ravel()
        out: List[str] = []
        for i in np.argsort(-scores):
            if int(i) == idx:
                continue
            out.append(str(df.iloc[int(i)]["title"]))
            if len(out) >= top_n:
                break
        return out
    except Exception:
        return []


# =========================
# HYBRID TF-IDF RECOMMENDER
# =========================
def _genre_text(genre_ids: List[int], names: Dict[int, str]) -> str:
    return " ".join(names.get(int(g), "") for g in genre_ids or [])


def _movie_text(m: dict, names: Dict[int, str]) -> str:
    title = m.get("title") or ""
    genres = _genre_text(m.get("genre_ids", []), names)
    return f"{title} {title} {genres} {genres} {m.get('overview') or ''}"


async def similar_movies(details: MovieDetails, limit: int = 12) -> List[MovieCard]:
    names = await get_genre_map()
    mid = details.tmdb_id
    genre_ids = [int(g["id"]) for g in details.genres]

    # ---- 1) candidate pool ----
    base = {"language": "en-US", "page": 1}
    calls = [
        safe_get(f"/movie/{mid}/recommendations", base),
        safe_get(f"/movie/{mid}/similar", base),
        safe_get(f"/movie/{mid}/keywords", {}),
    ]
    if genre_ids:
        disc = {"language": "en-US", "sort_by": "popularity.desc", "vote_count.gte": 100}
        calls += [
            safe_get("/discover/movie", {**disc, "with_genres": ",".join(map(str, genre_ids)), "page": 1}),
            safe_get("/discover/movie", {**disc, "with_genres": ",".join(map(str, genre_ids)), "page": 2}),
            safe_get("/discover/movie", {**disc, "with_genres": "|".join(map(str, genre_ids)), "page": 1}),
        ]

    responses = await asyncio.gather(*calls)
    keywords = [k.get("name", "") for k in responses[2].get("keywords", [])]

    pool: Dict[int, dict] = {}
    for resp in [responses[0], responses[1], *responses[3:]]:
        for m in resp.get("results", []):
            if m.get("id") and int(m["id"]) != mid:
                pool.setdefault(int(m["id"]), m)

    # your own TF-IDF model adds extra candidates when the title is in the dataset
    local_ids: Set[int] = set()
    titles = local_neighbour_titles(details.title, 10)
    if titles:
        found = await asyncio.gather(*(search_first(t) for t in titles))
        for m in found:
            if m and m.get("id") and int(m["id"]) != mid:
                pool.setdefault(int(m["id"]), m)
                local_ids.add(int(m["id"]))

    # only candidates that have a poster look good in the UI
    candidates = [m for m in pool.values() if m.get("poster_path")]
    if not candidates:
        return []

    # ---- 2) TF-IDF similarity ----
    try:
        gtext = " ".join(g["name"] for g in details.genres)
        kw = " ".join(keywords)
        query_text = (
            f"{details.title} {details.title} {gtext} {gtext} {kw} {kw} "
            f"{details.tagline or ''} {details.overview or ''}"
        )
        docs = [_movie_text(m, names) for m in candidates]

        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform([query_text] + docs)
        sims = linear_kernel(X[0:1], X[1:]).ravel()

        qset = set(genre_ids)
        scored = []
        for m, sim in zip(candidates, sims):
            cset = {int(g) for g in m.get("genre_ids", [])}
            overlap = len(qset & cset) / max(len(qset), 1)
            bonus = 1.0 if int(m["id"]) in local_ids else 0.0
            score = 0.65 * float(sim) + 0.30 * overlap + 0.05 * bonus
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [m for _, m in scored]
    except Exception as e:
        print(f"[WARN] TF-IDF ranking failed, using TMDB order: {e}")
        ranked = candidates

    return to_cards(ranked, limit)


# =========================
# FRONTEND
# =========================
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# =========================
# API ROUTES
# =========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/home", response_model=List[MovieCard])
async def home(
    category: str = Query("trending"),
    limit: int = Query(24, ge=1, le=50),
):
    if category == "trending":
        data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
    elif category in {"popular", "top_rated", "upcoming", "now_playing"}:
        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
    else:
        raise HTTPException(status_code=400, detail="Invalid category")
    return to_cards(data.get("results", []), limit)


@app.get("/api/search", response_model=List[MovieCard])
async def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(30, ge=1, le=50),
):
    data = await tmdb_get(
        "/search/movie",
        {"query": query, "include_adult": "false", "language": "en-US", "page": 1},
    )
    return to_cards(data.get("results", []), limit)


@app.get("/api/movie/{tmdb_id}", response_model=MovieDetails)
async def movie(tmdb_id: int):
    return await movie_details(tmdb_id)


@app.get("/api/movie/{tmdb_id}/recommendations", response_model=Recommendations)
async def recommendations(
    tmdb_id: int,
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    details = await movie_details(tmdb_id)

    try:
        similar = await similar_movies(details, tfidf_top_n)
    except Exception as e:
        print(f"[WARN] similar_movies failed: {e}")
        similar = []

    same_genre = await genre_cards(details, genre_limit)
    return Recommendations(tfidf=similar, genre=same_genre)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)