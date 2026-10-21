const app = document.getElementById("app");
const searchBox = document.getElementById("q");

const CATEGORIES = [
  ["trending", "🔥 Trending"],
  ["popular", "⭐ Popular"],
  ["top_rated", "🏆 Top Rated"],
  ["now_playing", "🎟️ Now Playing"],
  ["upcoming", "🗓️ Upcoming"],
];

const NO_POSTER =
  "data:image/svg+xml;utf8," +
  encodeURIComponent(
    "<svg xmlns='http://www.w3.org/2000/svg' width='500' height='750'><rect width='500' height='750' fill='#1b2030'/>" +
      "<text x='250' y='375' text-anchor='middle' fill='#8b93a7' font-family='Arial' font-size='28'>No poster</text></svg>"
  );

let category = "trending";
let searchTimer;

// ---------- helpers ----------
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function skeletons(n = 12) {
  return `<div class="grid">${'<div class="skeleton"></div>'.repeat(n)}</div>`;
}

function movieCard(m) {
  const year = (m.release_date || "").slice(0, 4);
  const rating = m.vote_average ? `<div class="rating">⭐ ${m.vote_average.toFixed(1)}</div>` : "";
  return `
    <div class="card" onclick="openMovie(${m.tmdb_id})">
      <div class="poster">
        <img loading="lazy" src="${m.poster_url || NO_POSTER}" alt="${escapeHTML(m.title)}"
             onerror="this.onerror=null;this.src='${NO_POSTER}'" />
        ${rating}
      </div>
      <div class="title">${escapeHTML(m.title)}</div>
      <div class="year">${year}</div>
    </div>`;
}

function movieGrid(list) {
  if (!list || list.length === 0) return `<div class="empty">No movies to show.</div>`;
  return `<div class="grid">${list.map(movieCard).join("")}</div>`;
}

// ---------- navigation (uses URL hash) ----------
function openMovie(id) {
  location.hash = `#/movie/${id}`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function goHome() {
  searchBox.value = "";
  location.hash = "#/";
}

function router() {
  const hash = location.hash;
  const movieMatch = hash.match(/^#\/movie\/(\d+)/);
  const searchMatch = hash.match(/^#\/search\/(.+)/);

  if (movieMatch) showDetails(movieMatch[1]);
  else if (searchMatch) showSearch(decodeURIComponent(searchMatch[1]));
  else showHome();
}

// ---------- pages ----------
async function showHome() {
  document.title = "CineMatch – Movie Recommender";
  const tabs = CATEGORIES.map(
    ([key, label]) => `<div class="tab ${key === category ? "active" : ""}" data-key="${key}">${label}</div>`
  ).join("");

  app.innerHTML = `<div class="tabs">${tabs}</div><div id="feed">${skeletons()}</div>`;

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      category = tab.dataset.key;
      showHome();
    };
  });

  try {
    const movies = await getJSON(`/api/home?category=${category}`);
    document.getElementById("feed").innerHTML = movieGrid(movies);
  } catch {
    document.getElementById("feed").innerHTML = `<div class="empty">Could not load movies.</div>`;
  }
}

async function showSearch(query) {
  app.innerHTML = `<h2 class="section">Results for “${escapeHTML(query)}”</h2><div id="feed">${skeletons()}</div>`;
  try {
    const movies = await getJSON(`/api/search?query=${encodeURIComponent(query)}`);
    document.getElementById("feed").innerHTML = movieGrid(movies);
  } catch {
    document.getElementById("feed").innerHTML = `<div class="empty">Search failed.</div>`;
  }
}

async function showDetails(id) {
  app.innerHTML = `
    <button class="back" onclick="history.back()">← Back</button>
    <div id="hero">${skeletons(1)}</div>
    <div id="recs"></div>`;

  // movie info
  try {
    const m = await getJSON(`/api/movie/${id}`);
    const year = (m.release_date || "").slice(0, 4);
    const hours = m.runtime ? `${Math.floor(m.runtime / 60)}h ${m.runtime % 60}m` : "";
    const genres = (m.genres || []).map((g) => `<span class="pill">${escapeHTML(g.name)}</span>`).join("");

    document.getElementById("hero").innerHTML = `
      <div class="hero">
        ${m.backdrop_url ? `<div class="bg" style="background-image:url('${m.backdrop_url}')"></div>` : ""}
        <div class="shade"></div>
        <img class="big" src="${m.poster_url || NO_POSTER}" alt="${escapeHTML(m.title)}"
             onerror="this.onerror=null;this.src='${NO_POSTER}'" />
        <div>
          <h1>${escapeHTML(m.title)} ${year ? `<span style="color:var(--muted);font-weight:500">(${year})</span>` : ""}</h1>
          ${m.tagline ? `<div class="tagline">${escapeHTML(m.tagline)}</div>` : ""}
          <div class="meta">
            ${m.vote_average ? `<span class="pill star">⭐ ${m.vote_average.toFixed(1)}</span>` : ""}
            ${hours ? `<span class="pill">${hours}</span>` : ""}
            ${genres}
          </div>
          <p class="overview">${escapeHTML(m.overview) || "No overview available."}</p>
        </div>
      </div>`;
    document.title = `${m.title} – CineMatch`;
  } catch {
    document.getElementById("hero").innerHTML = `<div class="empty">Could not load movie details.</div>`;
    return;
  }

  // recommendations
  const recs = document.getElementById("recs");
  recs.innerHTML = `<h2 class="section">🔎 Similar Movies</h2>${skeletons(6)}`;
  try {
    const data = await getJSON(`/api/movie/${id}/recommendations`);
    let html = "";
    if (data.tfidf.length) html += `<h2 class="section">🔎 Similar Movies</h2>${movieGrid(data.tfidf)}`;
    html += `<h2 class="section">🎭 More Like This (Genre)</h2>${movieGrid(data.genre)}`;
    recs.innerHTML = html;
  } catch {
    recs.innerHTML = `<div class="empty">No recommendations available right now.</div>`;
  }
}

// ---------- events ----------
document.getElementById("logo").onclick = goHome;

searchBox.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const value = searchBox.value.trim();
  searchTimer = setTimeout(() => {
    if (value.length >= 2) location.hash = `#/search/${encodeURIComponent(value)}`;
    else if (value.length === 0) location.hash = "#/";
  }, 400);
});

window.addEventListener("hashchange", router);
router();