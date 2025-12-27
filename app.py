import pickle
import streamlit as st
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LibriX",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

# ---------------- THEME COLORS ----------------
if dark_mode:
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    card_bg = "rgba(255,255,255,0.12)"
    accent = "#facc15"
else:
    bg_color = "#fffdf5"
    text_color = "#1f2937"
    card_bg = "rgba(255,255,255,0.65)"
    accent = "#f5d76e"

# ---------------- CUSTOM CSS ----------------
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    h1 {{
        text-align: center;
        font-weight: 800;
        letter-spacing: 1px;
        color: {text_color};
    }}

    label {{
        color: {text_color};
    }}

    /* Selectbox container */
    div[data-baseweb="select"] > div {{
        background-color: white;
        border-radius: 12px;
        border: 2px solid {accent};
    }}

    /* Selectbox input text */
    div[data-baseweb="select"] input {{
        color: black !important;
        font-weight: 500;
    }}

    /* Placeholder text */
    div[data-baseweb="select"] input::placeholder {{
        color: #6b7280 !important;
    }}

    /* Dropdown menu */
    ul[role="listbox"] {{
        background-color: yellow;
        color: black;
    }}

    /* Button */
    button[kind="primary"] {{
        background-color: {accent};
        color: white !important;
        border-radius: 12px;
        height: 3em;
        font-weight: 700;
        border: none;
    }}

    /* Glass Card */
    .book-card {{
        background: {card_bg};
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .book-card:hover {{
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 20px 45px rgba(0,0,0,0.25);
    }}

    .book-title {{
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 10px;
        color: {text_color};
    }}

    img {{
        border-radius: 14px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown("<h1>📚 LibriX</h1>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
model = pickle.load(open('artifacts/model.pkl','rb'))
book_names = pickle.load(open('artifacts/book_names.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))

# ---------------- FUNCTIONS ----------------
def fetch_poster(suggestion):
    poster_url = []
    for name in book_pivot.index[suggestion[0]]:
        idx = np.where(final_rating['Book-Title'] == name)[0][0]
        poster_url.append(final_rating.iloc[idx]['Image-URL-M'])
    return poster_url


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]

    _, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=6
    )

    books_list = list(book_pivot.index[suggestion[0]])
    poster_url = fetch_poster(suggestion)

    return books_list, poster_url

# ---------------- SEARCH (AUTOCOMPLETE) ----------------
selected_books = st.selectbox(
    "🔍 Search your favorite book",
    book_names,
    index=None,
    placeholder="Start typing a book name..."
)

# ---------------- RECOMMENDATION ----------------
if st.button("✨ Show Recommendation") and selected_books:
    recommended_books, poster_url = recommend_book(selected_books)

    cols = st.columns(5)

    for i, col in enumerate(cols, start=1):
        with col:
            st.markdown(
                f"""
                <div class="book-card">
                    <div class="book-title">{recommended_books[i]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.image(poster_url[i], use_container_width=True)
