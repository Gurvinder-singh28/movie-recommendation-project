import ast
import os
import pickle
import re
import time
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

try:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")
    if not TMDB_API_KEY:
        TMDB_API_KEY = st.secrets.get("TMDB_API_KEY")
except Exception:
    TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Add it to Streamlit Secrets "
        "or create a .env file with TMDB_API_KEY=your_key."
    )


# ============================================================
# ROBUST TMDB SESSION
# ============================================================

@st.cache_resource
def get_tmdb_session():
    session = requests.Session()

    retry_strategy = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=10,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            ),
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
    )

    return session


# ============================================================
# LOAD TRAINED ML ARTIFACTS
# ============================================================

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

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError("df.pkl must contain a pandas DataFrame.")

    required_columns = {"title", "genres", "vote_average", "popularity"}
    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"df.pkl is missing required columns: {sorted(missing)}"
        )

    title_to_idx = build_title_to_idx_map(indices_obj)

    return df, tfidf_matrix, tfidf_obj, title_to_idx


# ============================================================
# GENERAL HELPERS
# ============================================================

def _norm_title(title: str) -> str:
    return str(title).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    result = {}

    for key, value in indices.items():
        result[_norm_title(key)] = int(value)

    return result


def normalize_scores(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    finite = np.isfinite(values)
    cleaned = np.where(finite, values, 0.0)

    min_value = cleaned.min()
    max_value = cleaned.max()

    if max_value <= min_value:
        return np.zeros_like(cleaned)

    return (cleaned - min_value) / (max_value - min_value)


def parse_genres(value) -> set:
    """
    df['genres'] was converted in the notebook to a string such as:
    'Adventure Fantasy Family'

    This function also handles the original TMDB list/dict format,
    so the recommender remains robust.
    """
    if value is None:
        return set()

    if isinstance(value, float) and np.isnan(value):
        return set()

    if isinstance(value, (list, tuple)):
        names = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(str(name))
            else:
                names.append(str(item))
        return {x.strip().lower() for x in names if x.strip()}

    text = str(value).strip()

    if not text:
        return set()

    # Handle strings like:
    # "[{'id': 12, 'name': 'Adventure'}, ...]"
    if text.startswith("[") and "name" in text:
        try:
            parsed = ast.literal_eval(text)
            return parse_genres(parsed)
        except Exception:
            pass

    return {
        token.strip().lower()
        for token in text.split()
        if token.strip()
    }


def genre_jaccard(query_genres: set, candidate_genres: set) -> float:
    if not query_genres or not candidate_genres:
        return 0.0

    intersection = len(query_genres & candidate_genres)
    union = len(query_genres | candidate_genres)

    if union == 0:
        return 0.0

    return intersection / union


def preprocess_query_text(text: str) -> str:
    """
    The training notebook lowercases, removes punctuation,
    removes English stopwords and lemmatizes the training tags.

    The saved TfidfVectorizer already contains stop_words='english',
    so here we keep preprocessing lightweight and compatible with
    the saved vectorizer.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tmdb_tags(details: Dict) -> str:
    overview = details.get("overview") or ""
    tagline = details.get("tagline") or ""

    genres = details.get("genres") or []
    genre_names = []

    for genre in genres:
        if isinstance(genre, dict):
            name = genre.get("name")
            if name:
                genre_names.append(str(name))

    return preprocess_query_text(
        f"{overview} {' '.join(genre_names)} {tagline}"
    )


# ============================================================
# TMDB API
# ============================================================

def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    query = dict(params)
    query["api_key"] = TMDB_API_KEY

    url = f"{TMDB_BASE}{path}"
    session = get_tmdb_session()

    last_error = None

    for attempt in range(3):
        try:
            response = session.get(
                url,
                params=query,
                timeout=(10, 30),
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "2")

                try:
                    wait_seconds = min(int(retry_after), 10)
                except Exception:
                    wait_seconds = 2

                time.sleep(wait_seconds)
                continue

            if response.status_code != 200:
                raise RuntimeError(
                    f"TMDB error {response.status_code}: "
                    f"{response.text[:500]}"
                )

            return response.json()

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            last_error = exc

            if attempt < 2:
                time.sleep(2 ** attempt)
                continue

            break

        except requests.RequestException as exc:
            last_error = exc
            break

    raise RuntimeError(
        "TMDB connection failed after retries. "
        f"{type(last_error).__name__}: {last_error}"
    )


# ============================================================
# TMDB MOVIE FUNCTIONS
# ============================================================

def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20
) -> List[Dict]:
    cards = []

    for movie in (results or [])[:limit]:
        movie_id = movie.get("id")

        if not movie_id:
            continue

        cards.append(
            {
                "tmdb_id": int(movie_id),
                "title": movie.get("title") or movie.get("name") or "",
                "poster_url": make_img_url(movie.get("poster_path")),
                "release_date": movie.get("release_date") or "",
                "vote_average": movie.get("vote_average"),
            }
        )

    return cards


def tmdb_movie_details(movie_id: int) -> Dict:
    data = tmdb_get(
        f"/movie/{movie_id}",
        {"language": "en-US"},
    )

    return {
        "tmdb_id": int(data["id"]),
        "title": data.get("title") or "",
        "overview": data.get("overview") or "",
        "release_date": data.get("release_date") or "",
        "poster_url": make_img_url(data.get("poster_path")),
        "backdrop_url": make_img_url(data.get("backdrop_path")),
        "genres": data.get("genres") or [],
        "tagline": data.get("tagline") or "",
        "vote_average": data.get("vote_average") or 0.0,
        "vote_count": data.get("vote_count") or 0,
        "popularity": data.get("popularity") or 0.0,
    }


def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


def tmdb_search_first(query: str) -> Optional[dict]:
    data = tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


def attach_tmdb_card_by_title(title: str) -> Optional[Dict]:
    try:
        movie = tmdb_search_first(title)

        if not movie:
            return None

        return {
            "tmdb_id": int(movie["id"]),
            "title": movie.get("title") or title,
            "poster_url": make_img_url(movie.get("poster_path")),
            "release_date": movie.get("release_date") or "",
            "vote_average": movie.get("vote_average"),
        }

    except Exception:
        return None


# ============================================================
# CONTENT-BASED TF-IDF
# ============================================================

def get_local_idx_by_title(
    title: str,
    title_to_idx: Dict[str, int]
) -> int:
    key = _norm_title(title)

    if key not in title_to_idx:
        raise ValueError(
            f"Title not found in local dataset: '{title}'"
        )

    return int(title_to_idx[key])


def calculate_bayesian_rating(
    ratings: np.ndarray,
    vote_counts: np.ndarray,
    minimum_votes: float = 50.0,
) -> np.ndarray:
    """
    Bayesian-style rating estimate.

    This prevents a movie with 9.5/10 from only 3 votes
    from dominating a movie with thousands of votes.
    """
    ratings = np.asarray(ratings, dtype=float)
    vote_counts = np.asarray(vote_counts, dtype=float)

    valid_ratings = ratings[np.isfinite(ratings) & (ratings > 0)]

    global_mean = (
        float(valid_ratings.mean())
        if valid_ratings.size
        else 0.0
    )

    safe_votes = np.nan_to_num(
        vote_counts,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    safe_ratings = np.nan_to_num(
        ratings,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    weighted = (
        (safe_votes / (safe_votes + minimum_votes))
        * safe_ratings
        +
        (minimum_votes / (safe_votes + minimum_votes))
        * global_mean
    )

    return weighted


def hybrid_recommend_from_index(
    idx: int,
    top_n: int = 12,
) -> List[Dict]:
    """
    Hybrid recommendation for a movie already inside df.pkl.

    Final score:
        65% TF-IDF content similarity
        20% genre similarity
        10% Bayesian rating quality
         5% popularity

    TF-IDF remains the main signal, while the other signals
    improve ranking quality and reduce obviously weak matches.
    """
    df, tfidf_matrix, _, _ = load_resources()

    # --------------------------------------------------------
    # 1. TF-IDF similarity
    # --------------------------------------------------------
    query_vector = tfidf_matrix[idx]

    tfidf_scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    tfidf_scores = normalize_scores(tfidf_scores)

    # --------------------------------------------------------
    # 2. Genre similarity
    # --------------------------------------------------------
    query_genres = parse_genres(
        df.iloc[idx]["genres"]
    )

    genre_scores = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        candidate_genres = parse_genres(
            df.iloc[i]["genres"]
        )

        genre_scores[i] = genre_jaccard(
            query_genres,
            candidate_genres,
        )

    # --------------------------------------------------------
    # 3. Bayesian rating quality
    # --------------------------------------------------------
    ratings = pd.to_numeric(
        df["vote_average"],
        errors="coerce",
    ).fillna(0).to_numpy()

    if "vote_count" in df.columns:
        vote_counts = pd.to_numeric(
            df["vote_count"],
            errors="coerce",
        ).fillna(0).to_numpy()
    else:
        vote_counts = np.ones(len(df), dtype=float)

    bayesian_ratings = calculate_bayesian_rating(
        ratings,
        vote_counts,
    )

    rating_scores = np.clip(
        bayesian_ratings / 10.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # 4. Popularity
    # --------------------------------------------------------
    popularity = pd.to_numeric(
        df["popularity"],
        errors="coerce",
    ).fillna(0).to_numpy()

    popularity = np.log1p(
        np.maximum(popularity, 0)
    )

    popularity_scores = normalize_scores(
        popularity
    )

    # --------------------------------------------------------
    # 5. Final hybrid score
    # --------------------------------------------------------
    final_scores = (
        0.65 * tfidf_scores
        + 0.20 * genre_scores
        + 0.10 * rating_scores
        + 0.05 * popularity_scores
    )

    # --------------------------------------------------------
    # 6. Rank
    # --------------------------------------------------------
    order = np.argsort(-final_scores)

    query_title = _norm_title(
        df.iloc[idx]["title"]
    )

    recommendations = []

    for candidate_idx in order:
        candidate_idx = int(candidate_idx)

        if candidate_idx == idx:
            continue

        title = str(
            df.iloc[candidate_idx]["title"]
        )

        if _norm_title(title) == query_title:
            continue

        recommendations.append(
            {
                "title": title,
                "score": float(final_scores[candidate_idx]),
                "tfidf_score": float(tfidf_scores[candidate_idx]),
                "genre_score": float(genre_scores[candidate_idx]),
                "rating_score": float(rating_scores[candidate_idx]),
                "popularity_score": float(
                    popularity_scores[candidate_idx]
                ),
            }
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


def tfidf_recommend_from_details(
    details: Dict,
    top_n: int = 12,
) -> List[Dict]:
    """
    Cold-start recommendation.

    The movie can be a new TMDB movie that does not exist
    in the original training dataset.

    The saved TF-IDF vectorizer is reused with transform().
    """
    df, tfidf_matrix, tfidf_obj, _ = load_resources()

    query_tags = build_tmdb_tags(details)

    if not query_tags:
        return []

    # IMPORTANT:
    # Use transform(), NOT fit_transform().
    query_vector = tfidf_obj.transform(
        [query_tags]
    )

    tfidf_scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    tfidf_scores = normalize_scores(
        tfidf_scores
    )

    # --------------------------------------------------------
    # Genre signal
    # --------------------------------------------------------
    query_genres = {
        str(g.get("name", "")).strip().lower()
        for g in (details.get("genres") or [])
        if isinstance(g, dict) and g.get("name")
    }

    genre_scores = np.zeros(len(df), dtype=float)

    for i in range(len(df)):
        candidate_genres = parse_genres(
            df.iloc[i]["genres"]
        )

        genre_scores[i] = genre_jaccard(
            query_genres,
            candidate_genres,
        )

    # --------------------------------------------------------
    # Rating signal
    # --------------------------------------------------------
    ratings = pd.to_numeric(
        df["vote_average"],
        errors="coerce",
    ).fillna(0).to_numpy()

    if "vote_count" in df.columns:
        vote_counts = pd.to_numeric(
            df["vote_count"],
            errors="coerce",
        ).fillna(0).to_numpy()
    else:
        vote_counts = np.ones(len(df), dtype=float)

    bayesian_ratings = calculate_bayesian_rating(
        ratings,
        vote_counts,
    )

    rating_scores = np.clip(
        bayesian_ratings / 10.0,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Popularity signal
    # --------------------------------------------------------
    popularity = pd.to_numeric(
        df["popularity"],
        errors="coerce",
    ).fillna(0).to_numpy()

    popularity = np.log1p(
        np.maximum(popularity, 0)
    )

    popularity_scores = normalize_scores(
        popularity
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------
    final_scores = (
        0.65 * tfidf_scores
        + 0.20 * genre_scores
        + 0.10 * rating_scores
        + 0.05 * popularity_scores
    )

    order = np.argsort(
        -final_scores
    )

    query_title = _norm_title(
        details.get("title", "")
    )

    recommendations = []

    for i in order:
        i = int(i)

        title = str(
            df.iloc[i]["title"]
        )

        if _norm_title(title) == query_title:
            continue

        recommendations.append(
            {
                "title": title,
                "score": float(final_scores[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "genre_score": float(genre_scores[i]),
                "rating_score": float(rating_scores[i]),
                "popularity_score": float(
                    popularity_scores[i]
                ),
            }
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# ============================================================
# HOME FEED
# ============================================================

def get_home_feed(
    category: str = "popular",
    limit: int = 24,
) -> List[Dict]:

    if category == "trending":
        data = tmdb_get(
            "/trending/movie/day",
            {"language": "en-US"},
        )

        return tmdb_cards_from_results(
            data.get("results", []),
            limit=limit,
        )

    allowed = {
        "popular",
        "top_rated",
        "upcoming",
        "now_playing",
    }

    if category not in allowed:
        raise ValueError("Invalid category")

    data = tmdb_get(
        f"/movie/{category}",
        {
            "language": "en-US",
            "page": 1,
        },
    )

    return tmdb_cards_from_results(
        data.get("results", []),
        limit=limit,
    )


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def get_movie_details(
    tmdb_id: int
) -> Dict:
    return tmdb_movie_details(tmdb_id)


def get_genre_recommendations(
    tmdb_id: int,
    limit: int = 18,
) -> List[Dict]:

    details = tmdb_movie_details(
        tmdb_id
    )

    genres = details.get("genres") or []

    if not genres:
        return []

    # Use all genres instead of only the first genre.
    genre_ids = [
        str(g["id"])
        for g in genres
        if isinstance(g, dict) and g.get("id")
    ]

    discover = tmdb_get(
        "/discover/movie",
        {
            "with_genres": "|".join(genre_ids),
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )

    cards = tmdb_cards_from_results(
        discover.get("results", []),
        limit=limit + 5,
    )

    return [
        card
        for card in cards
        if card["tmdb_id"] != tmdb_id
    ][:limit]


def get_tfidf_recommendations(
    title: str,
    top_n: int = 12,
) -> List[Dict]:
    """
    Public compatibility function used by app.py.
    It now uses the hybrid ranking system.
    """
    _, _, _, title_to_idx = load_resources()

    try:
        idx = get_local_idx_by_title(
            title,
            title_to_idx,
        )

        return hybrid_recommend_from_index(
            idx,
            top_n=top_n,
        )

    except ValueError:
        # If title is not in the local dataset,
        # use TMDB to obtain metadata.
        movie = tmdb_search_first(title)

        if not movie:
            return []

        details = tmdb_movie_details(
            int(movie["id"])
        )

        return tfidf_recommend_from_details(
            details,
            top_n=top_n,
        )


# ============================================================
# SEARCH + RECOMMENDATION BUNDLE
# ============================================================

def search_bundle(
    query: str,
    tfidf_top_n: int = 12,
    genre_limit: int = 12,
) -> Dict:

    # --------------------------------------------------------
    # 1. Search TMDB
    # --------------------------------------------------------
    best = tmdb_search_first(
        query
    )

    if not best:
        raise ValueError(
            f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(
        best["id"]
    )

    # --------------------------------------------------------
    # 2. Full movie details
    # --------------------------------------------------------
    details = tmdb_movie_details(
        tmdb_id
    )

    # --------------------------------------------------------
    # 3. Hybrid recommendations
    # --------------------------------------------------------
    try:
        _, _, _, title_to_idx = load_resources()

        idx = get_local_idx_by_title(
            details["title"],
            title_to_idx,
        )

        recommendation_items = (
            hybrid_recommend_from_index(
                idx,
                top_n=tfidf_top_n,
            )
        )

    except Exception:
        # Cold-start path for new TMDB movies.
        recommendation_items = (
            tfidf_recommend_from_details(
                details,
                top_n=tfidf_top_n,
            )
        )

    # --------------------------------------------------------
    # 4. Get posters for recommended movies
    # --------------------------------------------------------
    recommendation_cards = []

    for item in recommendation_items:
        card = attach_tmdb_card_by_title(
            item["title"]
        )

        if card:
            recommendation_cards.append(
                {
                    **item,
                    "tmdb": card,
                }
            )

    # --------------------------------------------------------
    # 5. Genre fallback/secondary section
    # --------------------------------------------------------
    genre_recs = []

    if details.get("genres"):
        genre_ids = [
            str(g["id"])
            for g in details["genres"]
            if isinstance(g, dict) and g.get("id")
        ]

        if genre_ids:
            discover = tmdb_get(
                "/discover/movie",
                {
                    "with_genres": "|".join(genre_ids),
                    "language": "en-US",
                    "sort_by": "popularity.desc",
                    "page": 1,
                },
            )

            cards = tmdb_cards_from_results(
                discover.get("results", []),
                limit=genre_limit + 5,
            )

            genre_recs = [
                card
                for card in cards
                if card["tmdb_id"] != details["tmdb_id"]
            ][:genre_limit]

    return {
        "query": query,
        "movie_details": details,
        "tfidf_recommendations": recommendation_cards,
        "genre_recommendations": genre_recs,
    }