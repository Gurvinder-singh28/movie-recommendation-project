🎬 CineMatch — Movie Recommendation System

«A Machine Learning-powered movie discovery and recommendation platform built with Python, Scikit-learn, Streamlit and TMDB API.»

CineMatch helps users discover movies they may enjoy by combining content-based recommendation techniques with genre-based movie discovery and TMDB movie data.

🌐 Live Demo

🎬 "Try CineMatch" (https://cine-match-1j0b.onrender.com/?view=home)

💻 GitHub Repository

"View Source Code" (https://github.com/Gurvinder-singh28/movie-recommendation-project)

---

📌 Overview

With thousands of movies available across different platforms, finding something interesting to watch can be difficult.

CineMatch solves this problem by providing an interactive movie recommendation and discovery platform.

Users can search for movies, explore movie details and discover similar movies through a Machine Learning-based recommendation engine.

The project combines:

- Machine Learning
- Natural Language Processing concepts
- Content-based filtering
- TF-IDF vectorization
- Similarity-based recommendations
- TMDB API integration
- Streamlit application development

---

✨ Features

🎯 1. Content-Based Movie Recommendations

CineMatch uses TF-IDF-based representations to identify movies with similar content characteristics.

When a user selects a movie, the recommendation engine searches for movies with similar feature representations.

---

🎭 2. Genre-Based Movie Discovery

Users can discover additional movies based on genres associated with the selected movie.

This provides another recommendation path beyond direct content similarity.

---

🔎 3. Movie Search

Users can search for movies through the interactive application and select a movie to explore its information and recommendations.

---

🎬 4. TMDB API Integration

CineMatch integrates the TMDB API to retrieve movie-related information such as:

- Movie posters
- Movie titles
- Release dates
- Genres
- Ratings
- Movie descriptions
- Backdrop images
- Movie metadata

---

🔥 5. Movie Discovery Categories

The application provides multiple ways to discover movies, including:

- 🔥 Trending Movies
- ⭐ Popular Movies
- 🏆 Top Rated Movies
- 🎬 Now Playing Movies
- 🚀 Upcoming Movies

---

📖 6. Movie Details

Users can explore detailed information about a selected movie before deciding what to watch.

---

⚡ 7. Optimized Recommendation Resources

The project uses pre-generated and serialized Machine Learning resources such as the TF-IDF vectorizer, TF-IDF matrix and movie mappings.

This avoids rebuilding the recommendation resources every time the application starts.

---

🌐 8. Deployed Web Application

CineMatch is deployed as an online Streamlit application, allowing users to interact with the recommendation system directly from a browser.

---

🧠 How CineMatch Works

The recommendation workflow can be summarized as:

                    User
                     │
                     ▼
              Search / Select Movie
                     │
                     ▼
              Movie Information
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Content Similarity      Genre Discovery
          │                     │
          ▼                     ▼
    Similar Movies         Genre Movies
          │                     │
          └──────────┬──────────┘
                     ▼
             Recommendation UI

---

🔬 Recommendation Method

TF-IDF

TF-IDF (Term Frequency–Inverse Document Frequency) is used to transform textual movie features into numerical vectors.

This allows the system to represent movie information mathematically and compare movies based on their content.

Similarity

After converting the movie information into TF-IDF vectors, the system compares movies and identifies those with the highest similarity.

Conceptually:

Movie A → TF-IDF Vector
Movie B → TF-IDF Vector
Movie C → TF-IDF Vector
        ↓
Similarity Calculation
        ↓
Top Similar Movies

The highest-ranked similar movies are returned as recommendations.

---

🏗️ Project Architecture

                 ┌────────────────────┐
                 │      Streamlit     │
                 │     User Interface │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Recommendation     │
                 │ Engine             │
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       TF-IDF Resources             TMDB API
              │                         │
              ▼                         ▼
       Similar Movies              Movie Metadata
              │                         │
              └────────────┬────────────┘
                           ▼
                    CineMatch UI

---

🛠️ Tech Stack

Technology| Purpose
Python| Core development
Pandas| Data processing
NumPy| Numerical operations
Scikit-learn| Machine Learning
TF-IDF| Text feature extraction
Streamlit| Web application
TMDB API| Movie information
Pickle| Serialized ML resources
Jupyter Notebook| Data preprocessing and experimentation

---

📂 Project Structure

movie-recommendation-project/
│
├── app.py
│   └── Streamlit application
│
├── recommender.py
│   └── Recommendation logic and movie data handling
│
├── movies.ipynb
│   └── Data preprocessing and ML experimentation
│
├── df.pkl
│   └── Serialized movie dataframe
│
├── indices.pkl
│   └── Movie title/index mapping
│
├── tfidf.pkl
│   └── Serialized TF-IDF vectorizer
│
├── tfidf_matrix.pkl
│   └── Serialized TF-IDF matrix
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
    └── Git ignored files

---

⚙️ Installation

1. Clone the repository

git clone https://github.com/Gurvinder-singh28/movie-recommendation-project.git

2. Navigate to the project directory

cd movie-recommendation-project

3. Create a virtual environment

python -m venv venv

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

---

🔐 TMDB API Configuration

CineMatch uses the TMDB API for movie information.

Create an environment file:

.env

Add your API key:

TMDB_API_KEY=your_tmdb_api_key

«⚠️ Never commit your API key to GitHub.»

For deployment, configure the API key through the hosting platform's environment variables/secrets.

---

▶️ Run Locally

Start the Streamlit application with:

streamlit run app.py

The application will then be available locally in your browser.

---

📊 Machine Learning Pipeline

The project follows this general pipeline:

Movie Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Feature Preparation
     │
     ▼
TF-IDF Vectorization
     │
     ▼
TF-IDF Matrix
     │
     ▼
Similarity Calculation
     │
     ▼
Movie Recommendations
     │
     ▼
Streamlit Application

---

🎯 Learning Outcomes

Building CineMatch helped strengthen practical knowledge of:

- Machine Learning
- Recommendation Systems
- Content-Based Filtering
- NLP fundamentals
- TF-IDF Vectorization
- Similarity-based recommendations
- Data preprocessing
- Scikit-learn
- REST API integration
- Streamlit development
- Model/resource serialization
- ML application deployment

---

🚀 Future Improvements

Possible future enhancements:

- [ ] Hybrid recommendation system
- [ ] Collaborative filtering
- [ ] Personalized user profiles
- [ ] User rating system
- [ ] Watchlist functionality
- [ ] Advanced filtering by year, rating and language
- [ ] Recommendation explanations
- [ ] Improved ranking algorithm
- [ ] User authentication
- [ ] Movie trailers integration
- [ ] Personalized recommendations based on watch history

---

📈 Why This Project?

CineMatch was built to demonstrate how a Machine Learning recommendation algorithm can be transformed into a complete, interactive and deployed application.

Rather than keeping the ML model inside a notebook, this project focuses on taking the workflow from:

Data → Machine Learning → Recommendation Engine → UI → Deployment

---

👨‍💻 Author

Gurvinder Singh

AI/ML Engineer | Generative AI | Agentic AI | Python | Machine Learning | NLP

---

🔗 Project Links

🎬 Live Demo:
https://cine-match-1j0b.onrender.com/?view=home

💻 GitHub Repository:
https://github.com/Gurvinder-singh28/movie-recommendation-project

---

⭐ Show Your Support

If you find CineMatch useful or interesting, consider giving the repository a ⭐ on GitHub.

Built with Python, Machine Learning and a love for movies. 🎬🍿
