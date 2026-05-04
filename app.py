import os
import pickle
import sqlite3
from functools import wraps
from io import BytesIO
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from text_utils import clean_text
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MAX_INPUT_CHARS = 5000
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
NEWS_API_URL = "https://newsapi.org/v2/everything"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    )
}


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-this-secret-key")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def load_pickle_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def build_search_query(news_text):
    cleaned_words = clean_text(news_text).split()
    key_terms = []
    seen_terms = set()

    for word in cleaned_words:
        if word not in seen_terms and len(word) > 2:
            key_terms.append(word)
            seen_terms.add(word)
        if len(key_terms) == 8:
            break

    return " ".join(key_terms)


def load_model_objects():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        return load_pickle_file(MODEL_PATH), load_pickle_file(VECTORIZER_PATH)
    return None, None


def init_database():
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                news_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )

        existing_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(prediction_history)").fetchall()
        }
        if "user_id" not in existing_columns:
            cursor.execute("ALTER TABLE prediction_history ADD COLUMN user_id INTEGER")
        if "cleaned_text" not in existing_columns:
            cursor.execute(
                "ALTER TABLE prediction_history ADD COLUMN cleaned_text TEXT NOT NULL DEFAULT ''"
            )
        connection.commit()


def get_database_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def normalize_email(email):
    return (email or "").strip().lower()


def fetch_user_by_id(user_id):
    if not user_id:
        return None

    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, created_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def fetch_user_by_email(email):
    normalized_email = normalize_email(email)
    if not normalized_email:
        return None

    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, password_hash, created_at
            FROM users
            WHERE email = ?
            """,
            (normalized_email,),
        )
        row = cursor.fetchone()
    return dict(row) if row else None


def create_user(full_name, email, password):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO users (full_name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                full_name.strip(),
                normalize_email(email),
                generate_password_hash(password),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()
        return cursor.lastrowid


def login_required(api=False):
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            if g.user is None:
                if api:
                    return jsonify({"error": "Please log in to continue."}), 401
                return redirect(url_for("login"))
            return view_func(*args, **kwargs)

        return wrapped_view

    return decorator


def save_prediction(user_id, news_text, cleaned_text, prediction, confidence):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (
                user_id, news_text, cleaned_text, prediction, confidence, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                news_text,
                cleaned_text,
                prediction,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


def fetch_history(user_id, limit=10):
    try:
        safe_limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        safe_limit = 10
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, news_text, prediction, confidence, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, safe_limit),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


def fetch_history_stats(user_id):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) AS real_count,
                SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) AS fake_count
            FROM prediction_history
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return {"total_predictions": 0, "real_count": 0, "fake_count": 0}

    return {
        "total_predictions": row["total_predictions"] or 0,
        "real_count": row["real_count"] or 0,
        "fake_count": row["fake_count"] or 0,
    }


def clear_history_records(user_id):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
        connection.commit()


def fetch_article_text(url):
    if not url or url == "#":
        return ""

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return ""

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    paragraphs = [
        paragraph.get_text(" ", strip=True)
        for paragraph in soup.find_all("p")
        if paragraph.get_text(" ", strip=True)
    ]
    text = " ".join(paragraphs)
    return text[:4000]


def score_web_source(news_text, article):
    title = (article.get("title") or "").strip()
    description = (article.get("description") or "").strip()
    crawled_text = fetch_article_text(article.get("url") or "")
    combined_text = f"{title} {description} {crawled_text}".strip()

    news_words = set(clean_text(news_text).split())
    combined_words = set(clean_text(combined_text).split())
    overlap_count = len(news_words.intersection(combined_words))

    news_word_list = clean_text(news_text).split()
    phrase_hits = sum(1 for word in news_word_list[:12] if word in combined_words)
    support_score = min(100, overlap_count * 7 + phrase_hits * 3)

    return {
        "title": title or "Untitled article",
        "source_name": (article.get("source") or {}).get("name", "Unknown Source"),
        "url": article.get("url") or "#",
        "published_at": article.get("publishedAt") or "Unknown date",
        "support_score": support_score,
        "overlap_count": overlap_count,
    }


def validate_news_online(news_text):
    if not NEWS_API_KEY:
        return {
            "available": False,
            "real_score": None,
            "message": "Web verification is not configured. Add NEWS_API_KEY to enable internet checks.",
            "query": "",
            "sources": [],
        }

    query = build_search_query(news_text)
    if not query:
        return {
            "available": True,
            "real_score": None,
            "message": "Not enough meaningful text to search online.",
            "query": "",
            "sources": [],
        }

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return {
            "available": True,
            "real_score": None,
            "message": "Web verification failed. Please check your API key or network connection.",
            "query": query,
            "sources": [],
        }

    articles = payload.get("articles", [])
    sources = [score_web_source(news_text, article) for article in articles]
    sources.sort(key=lambda item: item["support_score"], reverse=True)

    top_sources = sources[:3]
    if top_sources:
        average_support = sum(item["support_score"] for item in top_sources) / len(top_sources)
        corroboration_bonus = min(20, len([item for item in top_sources if item["support_score"] >= 35]) * 6)
        real_score = round(min(100, average_support + corroboration_bonus), 2)
    else:
        real_score = 0.0

    if real_score >= 70:
        message = "Independent web checking found strong related reporting that supports this claim."
    elif real_score >= 45:
        message = "Independent web checking found limited or mixed support, so the claim should be reviewed carefully."
    else:
        message = "Independent web checking did not find strong supporting reporting for this claim."

    return {
        "available": True,
        "real_score": real_score,
        "message": message,
        "query": query,
        "sources": sources[:5],
    }


def build_combined_verification(predicted_label, confidence, web_validation):
    ai_real_score = confidence if predicted_label == "REAL" else 100 - confidence
    web_real_score = web_validation.get("real_score")

    if web_real_score is None:
        final_real_score = round(ai_real_score, 2)
        final_message = (
            "The final result is based mostly on the AI model because the app could not complete full web verification."
        )
    else:
        final_real_score = round(ai_real_score * 0.4 + web_real_score * 0.6, 2)
        if final_real_score >= 70:
            final_message = (
                "The final result comes from combining the AI model with independently checked online reporting, which together strongly support this claim."
            )
        elif final_real_score >= 45:
            final_message = (
                "The final result comes from combining the AI model with independently checked online reporting, but the evidence is mixed."
            )
        else:
            final_message = (
                "The final result comes from combining the AI model with independently checked online reporting, which did not strongly support this claim."
            )

    if final_real_score >= 65:
        final_verdict = "REAL"
    elif final_real_score <= 35:
        final_verdict = "FAKE"
    else:
        final_verdict = "UNCERTAIN"

    return {
        "final_verdict": final_verdict,
        "final_message": final_message,
        "final_confidence": round(max(final_real_score, 100 - final_real_score), 2),
        "sources": web_validation.get("sources", []),
        "query": web_validation.get("query", ""),
    }


def build_pdf_report(news_text, verification):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    body_style = styles["BodyText"]
    body_style.leading = 16
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    story = [
        Paragraph("AI-Based Fake News Detection Report", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small_style),
        Spacer(1, 0.15 * inch),
        Paragraph(f"<b>Final Verdict:</b> {verification.get('final_verdict', 'UNCERTAIN')}", body_style),
        Paragraph(
            f"<b>Final Confidence:</b> {verification.get('final_confidence', 0)}%",
            body_style,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Summary</b>", heading_style),
        Paragraph(verification.get("final_message", "No summary available."), body_style),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Search Query Used</b>", heading_style),
        Paragraph(verification.get("query", "Not available") or "Not available", body_style),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Submitted News Text</b>", heading_style),
        Paragraph(news_text.replace("\n", "<br/>"), body_style),
        Spacer(1, 0.2 * inch),
        Paragraph("<b>Checked Sources</b>", heading_style),
    ]

    sources = verification.get("sources", [])
    if sources:
        table_data = [["Source", "Support Score", "Published", "URL"]]
        for source in sources[:5]:
            table_data.append(
                [
                    source.get("source_name", "Unknown Source"),
                    str(source.get("support_score", "N/A")),
                    source.get("published_at", "Unknown date"),
                    source.get("url", "#"),
                ]
            )

        source_table = Table(table_data, colWidths=[1.2 * inch, 1.0 * inch, 1.5 * inch, 2.9 * inch])
        source_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F62FE")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6DEEB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FBFF")),
                ]
            )
        )
        story.append(source_table)
    else:
        story.append(Paragraph("No supporting sources were available for this report.", body_style))

    story.extend(
        [
            Spacer(1, 0.2 * inch),
            Paragraph("<b>Important Note</b>", heading_style),
            Paragraph(
                "This PDF is a decision-support report generated from machine learning analysis and web evidence. It should not be treated as absolute proof for high-stakes decisions.",
                small_style,
            ),
        ]
    )

    document.build(story)
    buffer.seek(0)
    return buffer


model, vectorizer = load_model_objects()
init_database()


@app.before_request
def load_logged_in_user():
    g.user = fetch_user_by_id(session.get("user_id"))


@app.route("/")
def home():
    if g.user is None:
        return redirect(url_for("login"))
    return render_template("index.html", user=g.user)


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if len(full_name) < 2:
            flash("Please enter your full name.", "error")
        elif "@" not in email or "." not in email:
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif fetch_user_by_email(email):
            flash("An account with that email already exists.", "error")
        else:
            try:
                create_user(full_name, email, password)
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
            else:
                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        user = fetch_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            flash("Welcome back.", "success")
            return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required()
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
@login_required(api=True)
def predict():
    global model, vectorizer

    if model is None or vectorizer is None:
        model, vectorizer = load_model_objects()

    if model is None or vectorizer is None:
        return (
            jsonify(
                {
                    "error": "Model files not found. Please run 'python train_model.py' first."
                }
            ),
            500,
        )

    data = request.get_json(silent=True) or {}
    news_text = (data.get("news_text") or "").strip()

    if not news_text:
        return jsonify({"error": "Please enter some news text."}), 400
    if len(news_text) > MAX_INPUT_CHARS:
        return jsonify({"error": f"News text must be under {MAX_INPUT_CHARS} characters."}), 400

    cleaned_text = clean_text(news_text)
    if len(cleaned_text.split()) < 3:
        return jsonify({"error": "Please enter a longer news statement for better prediction."}), 400

    text_vector = vectorizer.transform([cleaned_text])
    prediction_value = model.predict(text_vector)[0]
    probability_scores = model.predict_proba(text_vector)[0]

    predicted_label = "REAL" if prediction_value == 1 else "FAKE"
    confidence = round(max(probability_scores) * 100, 2)
    web_validation = validate_news_online(news_text)
    verification = build_combined_verification(predicted_label, confidence, web_validation)

    save_prediction(
        g.user["id"],
        news_text,
        cleaned_text,
        verification["final_verdict"],
        verification["final_confidence"],
    )

    return jsonify(
        {
            "verification": verification,
            "history": fetch_history(g.user["id"]),
            "stats": fetch_history_stats(g.user["id"]),
        }
    )


@app.route("/history", methods=["GET"])
@login_required(api=True)
def history():
    limit = request.args.get("limit", 20)
    return jsonify(
        {
            "items": fetch_history(g.user["id"], limit=limit),
            "stats": fetch_history_stats(g.user["id"]),
        }
    )


@app.route("/history/clear", methods=["POST"])
@login_required(api=True)
def clear_history():
    clear_history_records(g.user["id"])
    return jsonify(
        {
            "message": "Prediction history cleared successfully.",
            "items": [],
            "stats": fetch_history_stats(g.user["id"]),
        }
    )


@app.route("/report", methods=["POST"])
@login_required(api=True)
def download_report():
    data = request.get_json(silent=True) or {}
    news_text = (data.get("news_text") or "").strip()
    verification = data.get("verification") or {}

    if not news_text:
        return jsonify({"error": "News text is required to generate the report."}), 400
    if not verification:
        return jsonify({"error": "Verification data is required to generate the report."}), 400

    pdf_buffer = build_pdf_report(news_text, verification)
    filename = f"news_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/pdf",
    )


@app.route("/health", methods=["GET"])
def health():
    status = "ok" if model is not None and vectorizer is not None else "model_not_ready"
    return jsonify(
        {
            "status": status,
            "database": "ready",
            "model_loaded": model is not None,
            "news_api_configured": bool(NEWS_API_KEY),
        }
    ), 200


if __name__ == "__main__":
    init_database()
    app.run(debug=True, host="127.0.0.1", port=5000)
