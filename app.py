import os
from urllib.parse import quote

import streamlit as st

from recommender import (
    get_home_feed,
    get_movie_details,
    get_genre_recommendations,
    get_tfidf_recommendations,
    search_bundle,
    tmdb_search_movies,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

TMDB_IMG = "https://image.tmdb.org/t/p/w500"

POSTER_PLACEHOLDER = """
<svg xmlns="http://www.w3.org/2000/svg" width="500" height="750">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#151827"/>
      <stop offset="100%" stop-color="#090b12"/>
    </linearGradient>
  </defs>
  <rect width="500" height="750" fill="url(#g)"/>
  <text x="250" y="370" text-anchor="middle"
        font-family="Arial" font-size="30"
        fill="#8b93a7">NO POSTER</text>
</svg>
""".strip()

POSTER_PLACEHOLDER_URI = (
    "data:image/svg+xml;charset=UTF-8,"
    + quote(POSTER_PLACEHOLDER)
)


# ============================================================
# GLOBAL STYLES
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #090b12;
    --panel: #11141d;
    --panel-2: #151925;
    --border: rgba(255,255,255,.08);
    --text: #f4f5f7;
    --muted: #9299aa;
    --accent: #ff3d71;
    --accent-2: #ff6b3d;
}

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(255,61,113,.10), transparent 28rem),
        radial-gradient(circle at 10% 20%, rgba(88,101,242,.08), transparent 25rem),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding-top: 1.3rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: rgba(13,15,23,.96);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text);
}

h1, h2, h3, h4 {
    letter-spacing: -0.02em;
}

.hero {
    padding: 2.1rem 2.2rem;
    border: 1px solid var(--border);
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(255,61,113,.14), rgba(17,20,29,.90) 55%),
        var(--panel);
    box-shadow: 0 20px 70px rgba(0,0,0,.22);
    margin-bottom: 1.5rem;
}

.hero-kicker {
    color: #ff6b8e;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.hero-title {
    font-size: clamp(2.2rem, 5vw, 4.5rem);
    font-weight: 900;
    line-height: .98;
    margin: .35rem 0 .75rem 0;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    max-width: 780px;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    margin: 1.4rem 0 .75rem 0;
}

.section-subtitle {
    color: var(--muted);
    font-size: .9rem;
    margin-top: -.45rem;
    margin-bottom: 1rem;
}

.movie-card {
    background: linear-gradient(
        180deg,
        rgba(255,255,255,.045),
        rgba(255,255,255,.018)
    );
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 9px;
    transition: transform .18s ease, border-color .18s ease;
    min-height: 100%;
}

.movie-card:hover {
    transform: translateY(-4px);
    border-color: rgba(255,61,113,.35);
}

.poster {
    width: 100%;
    aspect-ratio: 2 / 3;
    object-fit: cover;
    border-radius: 13px;
    display: block;
    background: #151925;
}

.movie-name {
    color: #f3f4f6;
    font-weight: 750;
    font-size: .91rem;
    line-height: 1.2;
    min-height: 2.25rem;
    margin: .6rem .15rem .25rem;
}

.movie-meta {
    color: var(--muted);
    font-size: .76rem;
    margin: .1rem .15rem .35rem;
}

.match {
    display: inline-block;
    background: rgba(255,61,113,.13);
    border: 1px solid rgba(255,61,113,.22);
    color: #ff8ca8;
    border-radius: 999px;
    padding: .18rem .45rem;
    font-size: .69rem;
    font-weight: 800;
}

.rating {
    display: inline-block;
    background: rgba(255,193,7,.10);
    border: 1px solid rgba(255,193,7,.18);
    color: #ffd76a;
    border-radius: 999px;
    padding: .18rem .45rem;
    font-size: .69rem;
    font-weight: 800;
}

.detail-panel {
    background: rgba(17,20,29,.88);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 1.35rem;
    box-shadow: 0 20px 70px rgba(0,0,0,.18);
}

.detail-title {
    font-size: clamp(2rem, 4vw, 3.7rem);
    font-weight: 900;
    line-height: 1;
    margin: .25rem 0 .8rem;
}

.detail-tagline {
    color: #b8bfce;
    font-style: italic;
    margin-bottom: 1rem;
}

.stat {
    background: rgba(255,255,255,.035);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: .7rem .8rem;
}

.stat-label {
    color: var(--muted);
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.stat-value {
    color: white;
    font-weight: 800;
    margin-top: .15rem;
}

.backdrop {
    width: 100%;
    max-height: 390px;
    object-fit: cover;
    border-radius: 20px;
    opacity: .88;
    border: 1px solid var(--border);
}

.info-strip {
    color: #b5bccb;
    font-size: .82rem;
    padding: .8rem 1rem;
    background: rgba(255,255,255,.025);
    border: 1px solid var(--border);
    border-radius: 13px;
    margin: .8rem 0 1.2rem;
}

div.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,.10);
    background: rgba(255,255,255,.045);
    color: #f5f6f8;
    font-weight: 700;
    width: 100%;
    transition: all .18s ease;
}

div.stButton > button:hover {
    border-color: rgba(255,61,113,.55);
    color: white;
    background: rgba(255,61,113,.10);
}

.stTextInput input {
    border-radius: 14px;
    background: #11141d;
    border: 1px solid rgba(255,255,255,.10);
    color: white;
    padding: .9rem 1rem;
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
    background: #11141d;
}

hr {
    border-color: var(--border);
}

.small-muted {
    color: var(--muted);
    font-size: .86rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE / ROUTING
# ============================================================

if "view" not in st.session_state:
    st.session_state.view = "home"

if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None


qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")

if qp_view in ("home", "details"):
    st.session_state.view = qp_view

if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except ValueError:
        pass


def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None
    st.query_params["view"] = "home"

    if "id" in st.query_params:
        del st.query_params["id"]

    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# ============================================================
# DATA HELPERS
# ============================================================

@st.cache_data(ttl=60, show_spinner=False)
def safe_call(fn_name: str, **kwargs):
    fn_map = {
        "home": get_home_feed,
        "search": tmdb_search_movies,
        "details": get_movie_details,
        "bundle": search_bundle,
        "genre": get_genre_recommendations,
        "tfidf": get_tfidf_recommendations,
    }

    try:
        return fn_map[fn_name](**kwargs), None
    except Exception as exc:
        return None, str(exc)


def poster_html(src, title="Movie poster"):
    safe_src = src or POSTER_PLACEHOLDER_URI
    safe_alt = (title or "Movie poster").replace('"', "")

    return f"""
    <img
        class="poster"
        src="{safe_src}"
        alt="{safe_alt}"
        onerror="this.onerror=null;this.src='{POSTER_PLACEHOLDER_URI}';"
    />
    """


def poster_grid(
    cards,
    cols=5,
    key_prefix="grid",
    show_match=True,
):
    if not cards:
        st.info("No movies available right now.")
        return

    rows = (len(cards) + cols - 1) // cols
    index = 0

    for row in range(rows):
        columns = st.columns(cols, gap="small")

        for col_idx in range(cols):
            if index >= len(cards):
                break

            movie = cards[index]
            index += 1

            tmdb_id = movie.get("tmdb_id")
            title = movie.get("title") or "Untitled"
            poster = movie.get("poster_url")

            rating = movie.get("vote_average")
            score = movie.get("score")

            with columns[col_idx]:
                # Single-line string prevents Streamlit from rendering HTML as a code block
                card_html = f'<div class="movie-card">{poster_html(poster, title)}<div class="movie-name">{title}</div></div>'
                st.markdown(card_html, unsafe_allow_html=True)

                badges = []

                if rating is not None:
                    try:
                        badges.append(
                            f'<span class="rating">★ {float(rating):.1f}</span>'
                        )
                    except (TypeError, ValueError):
                        pass

                if show_match and score is not None:
                    try:
                        match_percent = min(
                            99,
                            max(1, round(float(score) * 100)),
                        )
                        badges.append(
                            f'<span class="match">MATCH {match_percent}%</span>'
                        )
                    except (TypeError, ValueError):
                        pass

                if badges:
                    st.markdown(
                        " ".join(badges),
                        unsafe_allow_html=True,
                    )

                if st.button(
                    "View movie",
                    key=f"{key_prefix}_{row}_{col_idx}_{index}_{tmdb_id}",
                ):
                    if tmdb_id:
                        goto_details(tmdb_id)


def to_cards_from_tfidf_items(items):
    cards = []

    for item in items or []:
        tmdb = item.get("tmdb") or {}

        if not tmdb.get("tmdb_id"):
            continue

        cards.append(
            {
                "tmdb_id": tmdb["tmdb_id"],
                "title": tmdb.get("title")
                or item.get("title")
                or "Untitled",
                "poster_url": tmdb.get("poster_url"),
                "vote_average": tmdb.get("vote_average"),
                "score": item.get("score"),
                "tfidf_score": item.get("tfidf_score"),
                "genre_score": item.get("genre_score"),
            }
        )

    return cards


def parse_search_results(data, keyword, limit=20):
    if not isinstance(data, dict):
        return [], []

    raw = data.get("results") or []
    keyword = keyword.strip().lower()

    cards = []

    for movie in raw:
        title = (movie.get("title") or "").strip()
        tmdb_id = movie.get("id")

        if not title or not tmdb_id:
            continue

        cards.append(
            {
                "tmdb_id": int(tmdb_id),
                "title": title,
                "poster_url": (
                    f"{TMDB_IMG}{movie['poster_path']}"
                    if movie.get("poster_path")
                    else None
                ),
                "release_date": movie.get("release_date") or "",
                "vote_average": movie.get("vote_average"),
            }
        )

    matched = [
        card
        for card in cards
        if keyword in card["title"].lower()
    ]

    final_cards = matched if matched else cards

    suggestions = []

    for movie in final_cards[:10]:
        year = (movie.get("release_date") or "")[:4]

        label = (
            f"{movie['title']} ({year})"
            if year
            else movie["title"]
        )

        suggestions.append(
            (label, movie["tmdb_id"])
        )

    return suggestions, final_cards[:limit]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("AI-powered movie discovery")

    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.divider()

    st.markdown("### Discover")

    home_category = st.selectbox(
        "Browse",
        [
            "trending",
            "popular",
            "top_rated",
            "now_playing",
            "upcoming",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    grid_cols = st.slider(
        "Movies per row",
        min_value=4,
        max_value=6,
        value=5,
    )

    st.divider()

    st.markdown("### 🧠 Recommendation Engine")
    st.caption(
        "Hybrid ranking: TF-IDF + genre similarity + "
        "rating quality + popularity."
    )


# ============================================================
# HOME VIEW
# ============================================================

if st.session_state.view == "home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Machine Learning • TMDB • Streamlit</div>
            <div class="hero-title">Find your next<br>favorite movie.</div>
            <div class="hero-subtitle">
                Search any movie and discover content-similar recommendations
                powered by a hybrid ML ranking system.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔎 Search movies")

    typed = st.text_input(
        "Movie search",
        placeholder="Try: Interstellar, Batman, Avengers, Inception...",
        label_visibility="collapsed",
    )

    if typed.strip():

        if len(typed.strip()) < 2:
            st.caption("Type at least 2 characters.")

        else:
            data, error = safe_call(
                "search",
                query=typed.strip(),
            )

            if error:
                st.error(f"Search failed: {error}")

            else:
                suggestions, cards = parse_search_results(
                    data,
                    typed.strip(),
                    limit=20,
                )

                if suggestions:
                    labels = [
                        "-- Select a movie --"
                    ] + [
                        suggestion[0]
                        for suggestion in suggestions
                    ]

                    selected = st.selectbox(
                        "Choose a movie",
                        labels,
                        label_visibility="collapsed",
                    )

                    if selected != "-- Select a movie --":
                        lookup = {
                            label: tmdb_id
                            for label, tmdb_id in suggestions
                        }

                        goto_details(
                            lookup[selected]
                        )

                st.markdown(
                    '<div class="section-title">Search results</div>',
                    unsafe_allow_html=True,
                )

                poster_grid(
                    cards,
                    cols=grid_cols,
                    key_prefix="search",
                    show_match=False,
                )

        st.stop()

    # --------------------------------------------------------
    # HOME FEED
    # --------------------------------------------------------

    category_title = home_category.replace(
        "_", " "
    ).title()

    st.markdown(
        f'<div class="section-title">🔥 {category_title}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Fresh movies from TMDB"
        "</div>",
        unsafe_allow_html=True,
    )

    home_cards, error = safe_call(
        "home",
        category=home_category,
        limit=24,
    )

    if error:
        st.error(
            "Movie feed could not be loaded. "
            f"{error}"
        )
        st.stop()

    poster_grid(
        home_cards,
        cols=grid_cols,
        key_prefix="home",
        show_match=False,
    )


# ============================================================
# DETAILS VIEW
# ============================================================

# ============================================================
# DETAILS VIEW
# ============================================================

elif st.session_state.view == "details":

    tmdb_id = st.session_state.selected_tmdb_id

    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back to Home"):
            goto_home()
        st.stop()

    data, error = safe_call(
        "details",
        tmdb_id=tmdb_id,
    )

    if error or not data:
        st.error(
            f"Could not load movie details: "
            f"{error or 'Unknown error'}"
        )

        if st.button("← Back to Home"):
            goto_home()

        st.stop()

    if st.button("← Back to Home"):
        goto_home()

    title = data.get("title") or "Untitled"
    poster = data.get("poster_url")
    backdrop = data.get("backdrop_url")
    overview = data.get("overview") or "No overview available."
    tagline = data.get("tagline") or ""

    release_date = data.get("release_date") or "Unknown"

    genres = ", ".join(
        genre.get("name", "")
        for genre in data.get("genres", [])
        if isinstance(genre, dict)
    ) or "Unknown"

    rating = data.get("vote_average") or 0
    vote_count = data.get("vote_count") or 0

    # --------------------------------------------------------
    # Backdrop
    # --------------------------------------------------------

    if backdrop:
        st.markdown(
            f'<img class="backdrop" src="{backdrop}" alt="{title} backdrop">',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # Movie hero/details
    # --------------------------------------------------------

    left, right = st.columns(
        [1, 2.25],
        gap="large",
    )

    with left:
        st.markdown(
            f'<div class="detail-panel">{poster_html(poster, title)}</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f'''<div class="detail-panel">
<div class="hero-kicker">MOVIE DETAILS</div>
<div class="detail-title">{title}</div>
<div class="detail-tagline">{tagline}</div>
<div class="info-strip"><b>Genres:</b> {genres}</div>
</div>''',
            unsafe_allow_html=True,
        )

        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.markdown(
                f'<div class="stat"><div class="stat-label">TMDB Rating</div><div class="stat-value">★ {float(rating):.1f}/10</div></div>',
                unsafe_allow_html=True,
            )

        with stat2:
            st.markdown(
                f'<div class="stat"><div class="stat-label">Release</div><div class="stat-value">{release_date}</div></div>',
                unsafe_allow_html=True,
            )

        with stat3:
            st.markdown(
                f'<div class="stat"><div class="stat-label">Votes</div><div class="stat-value">{int(vote_count):,}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### 📖 Overview")
    st.write(overview)

    st.divider()

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🧠 Recommended for you</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        "Ranked using TF-IDF content similarity, genre overlap, "
        "rating quality and popularity."
        "</div>",
        unsafe_allow_html=True,
    )

    bundle, bundle_error = safe_call(
        "bundle",
        query=title,
        tfidf_top_n=12,
        genre_limit=12,
    )

    if bundle_error or not bundle:
        st.warning(
            "The ML recommendation service is temporarily "
            "unavailable. Showing genre recommendations instead."
        )

        genre_only, genre_error = safe_call(
            "genre",
            tmdb_id=tmdb_id,
            limit=12,
        )

        if not genre_error:
            poster_grid(
                genre_only,
                cols=grid_cols,
                key_prefix="genre_fallback",
                show_match=False,
            )

        st.stop()

    # --------------------------------------------------------
    # Hybrid / TF-IDF recommendations
    # --------------------------------------------------------

    hybrid_cards = to_cards_from_tfidf_items(
        bundle.get(
            "tfidf_recommendations"
        )
    )

    st.markdown(
        "#### 🎯 Top Similar Movies"
    )

    if hybrid_cards:
        poster_grid(
            hybrid_cards,
            cols=grid_cols,
            key_prefix="hybrid",
            show_match=True,
        )
    else:
        st.info(
            "No content-similar movies were found."
        )

    # --------------------------------------------------------
    # Genre recommendations
    # --------------------------------------------------------

    st.markdown(
        "#### 🎭 More Like This"
    )

    st.caption(
        "Additional movies sharing the selected movie's genres."
    )

    genre_cards = bundle.get(
        "genre_recommendations",
        [],
    )

    poster_grid(
        genre_cards,
        cols=grid_cols,
        key_prefix="genre",
        show_match=False,
    )