import os
import pickle
import re
import sqlite3
from collections import Counter
from functools import wraps
from io import BytesIO
from datetime import datetime
from urllib.parse import urlparse

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
from comparison_utils import compute_similarity_metrics, extract_comparison_features
from text_utils import STOP_WORDS, clean_text
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MAX_INPUT_CHARS = 5000
EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
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


def extract_major_keywords(news_text, max_terms=8):
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(news_text))
    lowered_tokens = [token.lower() for token in raw_tokens]
    filtered_tokens = [
        token for token in lowered_tokens if token not in STOP_WORDS and len(token) > 2
    ]
    if not filtered_tokens:
        return []

    token_counts = Counter(filtered_tokens)
    phrase_counts = Counter()
    for size in (2, 3):
        for index in range(len(filtered_tokens) - size + 1):
            phrase = " ".join(filtered_tokens[index : index + size])
            phrase_counts[phrase] += 1

    scored_terms = []
    seen_terms = set()

    for index, token in enumerate(filtered_tokens):
        if token in seen_terms:
            continue
        position_weight = 1.0 if index < 12 else 0.65
        score = token_counts[token] * 2.5 + position_weight
        scored_terms.append((score, token))
        seen_terms.add(token)

    for phrase, count in phrase_counts.items():
        first_word = phrase.split()[0]
        position_weight = 1.4 if filtered_tokens.index(first_word) < 12 else 0.9
        score = count * (3.0 + len(phrase.split()) * 0.6) + position_weight
        scored_terms.append((score, phrase))

    scored_terms.sort(key=lambda item: (-item[0], len(item[1].split()), item[1]))

    keywords = []
    used_words = set()
    for _, term in scored_terms:
        term_words = term.split()
        if len(keywords) >= max_terms:
            break
        if len(term_words) > 1:
            keywords.append(term)
            used_words.update(term_words)
            continue
        if term not in used_words:
            keywords.append(term)
            used_words.add(term)

    return keywords[:max_terms]


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

    keywords = extract_major_keywords(news_text)
    return {
        "keywords": keywords,
        "query": " ".join(key_terms),
    }


def load_model_objects():
    if os.path.exists(MODEL_PATH):
        return load_pickle_file(MODEL_PATH)
    return None


def initialize_database_schema(connection):
    cursor = connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        cursor.execute("PRAGMA journal_mode=DELETE")
        cursor.execute("PRAGMA synchronous=FULL")
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
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL,
            subject TEXT NOT NULL,
            message TEXT NOT NULL,
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


def cleanup_sqlite_sidecar_files():
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar_path = f"{DB_PATH}{suffix}"
        if os.path.exists(sidecar_path):
            try:
                os.remove(sidecar_path)
            except OSError:
                continue
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) == 0:
        try:
            os.remove(DB_PATH)
        except OSError:
            pass


def init_database():
    try:
        with sqlite3.connect(DB_PATH) as connection:
            initialize_database_schema(connection)
    except sqlite3.OperationalError as error:
        if "disk I/O error" not in str(error):
            raise
        cleanup_sqlite_sidecar_files()
        with sqlite3.connect(DB_PATH) as connection:
            initialize_database_schema(connection)


def get_database_connection():
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def normalize_email(email):
    return (email or "").strip().lower()


def is_valid_email(email):
    return bool(EMAIL_PATTERN.match(normalize_email(email)))


def get_admin_emails():
    configured_value = os.getenv("ADMIN_EMAILS", "")
    return {
        normalize_email(email)
        for email in configured_value.split(",")
        if normalize_email(email)
    }


def is_admin_user(user):
    return bool(user and normalize_email(user.get("email")) in get_admin_emails())


def get_safe_redirect_target(default_endpoint="dashboard"):
    next_value = (request.args.get("next") or request.form.get("next") or "").strip()
    if not next_value:
        return url_for(default_endpoint)

    parsed_url = urlparse(next_value)
    if parsed_url.scheme or parsed_url.netloc:
        return url_for(default_endpoint)
    if not next_value.startswith("/"):
        return url_for(default_endpoint)
    return next_value


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
                return redirect(url_for("login", next=request.path))
            return view_func(*args, **kwargs)

        return wrapped_view

    return decorator


def admin_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if g.user is None:
            return redirect(url_for("login", next=request.path))
        if not g.is_admin:
            flash("You do not have access to the admin feedback panel.", "error")
            return redirect(url_for("dashboard"))
        return view_func(*args, **kwargs)

    return wrapped_view


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
                SUM(CASE WHEN prediction IN ('REAL', 'SUPPORTED') THEN 1 ELSE 0 END) AS real_count,
                SUM(CASE WHEN prediction IN ('UNCERTAIN', 'NEEDS REVIEW') THEN 1 ELSE 0 END) AS uncertain_count,
                SUM(CASE WHEN prediction IN ('FAKE', 'NOT SUPPORTED') THEN 1 ELSE 0 END) AS fake_count
            FROM prediction_history
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = cursor.fetchone()

    if row is None:
        return {
            "total_predictions": 0,
            "real_count": 0,
            "uncertain_count": 0,
            "fake_count": 0,
        }

    return {
        "total_predictions": row["total_predictions"] or 0,
        "real_count": row["real_count"] or 0,
        "uncertain_count": row["uncertain_count"] or 0,
        "fake_count": row["fake_count"] or 0,
    }


def clear_history_records(user_id):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
        connection.commit()


def save_feedback(user, subject, message):
    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (
                user_id, full_name, email, subject, message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["full_name"],
                user["email"],
                subject.strip(),
                message.strip(),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


def fetch_feedback_items(limit=100):
    try:
        safe_limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        safe_limit = 100

    with get_database_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT id, full_name, email, subject, message, created_at
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        )
        rows = cursor.fetchall()
    return [dict(row) for row in rows]


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
    similarity_metrics = compute_similarity_metrics(news_text, combined_text)
    if model is None:
        support_probability = 0.0
    else:
        feature_vector = [extract_comparison_features(news_text, combined_text)]
        support_probability = float(model.predict_proba(feature_vector)[0][1])

    semantic_score = (
        support_probability * 0.48
        + similarity_metrics["claim_coverage"] * 0.20
        + similarity_metrics["bigram_score"] * 0.12
        + similarity_metrics["sequence_score"] * 0.10
        + similarity_metrics["leading_term_score"] * 0.10
    )
    support_score = round(min(100, semantic_score * 100), 2)

    return {
        "title": title or "Untitled article",
        "source_name": (article.get("source") or {}).get("name", "Unknown Source"),
        "url": article.get("url") or "#",
        "published_at": article.get("publishedAt") or "Unknown date",
        "support_score": support_score,
        "overlap_count": similarity_metrics["overlap_count"],
        "semantic_score": round(similarity_metrics["sequence_score"] * 100, 2),
        "keyword_coverage": round(similarity_metrics["claim_coverage"] * 100, 2),
        "phrase_match": round(similarity_metrics["bigram_score"] * 100, 2),
        "evidence_excerpt": (combined_text[:280] + "...") if len(combined_text) > 280 else combined_text,
    }


def validate_news_online(news_text):
    if not NEWS_API_KEY:
        return {
            "available": False,
            "support_score": None,
            "message": "News API verification is not configured. Add NEWS_API_KEY to compare with live reporting.",
            "query": "",
            "sources": [],
            "supportive_source_count": 0,
        }

    query_payload = build_search_query(news_text)
    query = query_payload["query"]
    keywords = query_payload["keywords"]
    if not query:
        return {
            "available": True,
            "support_score": None,
            "message": "Not enough meaningful text to search News API results.",
            "query": "",
            "sources": [],
            "supportive_source_count": 0,
            "keywords": [],
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
            "support_score": None,
            "message": "News API verification failed. Please check your API key or network connection.",
            "query": query,
            "sources": [],
            "supportive_source_count": 0,
            "keywords": keywords,
        }

    articles = payload.get("articles", [])
    sources = [score_web_source(news_text, article) for article in articles]
    sources = [
        source
        for source in sources
        if source["support_score"] > 0 or source["overlap_count"] > 0
    ]
    sources.sort(key=lambda item: item["support_score"], reverse=True)

    top_sources = sources[:3]
    supportive_source_count = len([item for item in top_sources if item["support_score"] >= 60])
    if top_sources:
        best_score = top_sources[0]["support_score"]
        average_support = sum(item["support_score"] for item in top_sources) / len(top_sources)
        corroboration_bonus = min(12, supportive_source_count * 4)
        support_score = round(min(100, best_score * 0.55 + average_support * 0.35 + corroboration_bonus), 2)
    else:
        support_score = 0.0

    if support_score >= 70 and supportive_source_count >= 2:
        message = "News API results contain multiple close matches that strongly support the submitted news text."
    elif support_score >= 40:
        message = "News API results show some overlap, but the match is not strong enough to treat as reliable confirmation."
    else:
        message = "News API results did not provide strong supporting coverage for the submitted news text."

    return {
        "available": True,
        "support_score": support_score,
        "message": message,
        "query": query,
        "sources": sources[:5],
        "supportive_source_count": supportive_source_count,
        "keywords": keywords,
    }


def build_comparison_verification(web_validation):
    support_score = web_validation.get("support_score")

    if support_score is None:
        final_verdict = "UNCERTAIN"
        match_score = 0.0
        final_message = web_validation.get("message") or (
            "The app could not compare the submitted text against News API results."
        )
    elif support_score > 60:
        final_verdict = "REAL"
        match_score = round(support_score, 2)
        final_message = (
            "The submitted news text achieved an evidence match score above 60%, so the result is positive and the content is likely real."
        )
    elif support_score <= 32:
        final_verdict = "FAKE"
        match_score = round(support_score, 2)
        final_message = (
            "The submitted news text does not meaningfully match the retrieved News API reporting, so it is likely fake."
        )
    else:
        final_verdict = "UNCERTAIN"
        match_score = round(support_score, 2)
        final_message = (
            "The submitted news text has only partial semantic overlap with the News API results, so the truth value is uncertain."
        )

    return {
        "final_verdict": final_verdict,
        "final_message": final_message,
        "final_confidence": match_score,
        "match_score": match_score,
        "sources": web_validation.get("sources", []),
        "query": web_validation.get("query", ""),
        "keywords": web_validation.get("keywords", []),
        "analysis_basis": "Major keywords were extracted from the entered text, News API was searched with those keywords, and the returned articles were compared semantically against the full content.",
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
            f"<b>Evidence Match Score:</b> {verification.get('match_score', verification.get('final_confidence', 0))}%",
            body_style,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Summary</b>", heading_style),
        Paragraph(verification.get("final_message", "No summary available."), body_style),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Analysis Basis</b>", heading_style),
        Paragraph(
            verification.get("analysis_basis", "Compared against retrieved News API articles."),
            body_style,
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Search Query Used</b>", heading_style),
        Paragraph(verification.get("query", "Not available") or "Not available", body_style),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Major Keywords Used</b>", heading_style),
        Paragraph(", ".join(verification.get("keywords", [])) or "Not available", body_style),
        Spacer(1, 0.15 * inch),
        Paragraph("<b>Submitted News Text</b>", heading_style),
        Paragraph(news_text.replace("\n", "<br/>"), body_style),
        Spacer(1, 0.2 * inch),
        Paragraph("<b>Checked Sources</b>", heading_style),
    ]

    sources = verification.get("sources", [])
    if sources:
        table_data = [["Source", "Truth Score", "Published", "URL"]]
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
                "This PDF summarizes how closely the submitted text matched News API coverage. It is a support signal, not absolute proof of truth.",
                small_style,
            ),
        ]
    )

    document.build(story)
    buffer.seek(0)
    return buffer


model = load_model_objects()
init_database()


@app.before_request
def load_logged_in_user():
    g.user = fetch_user_by_id(session.get("user_id"))
    g.is_admin = is_admin_user(g.user)


@app.route("/")
def home():
    if g.user is not None:
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/dashboard")
@login_required()
def dashboard():
    stats = fetch_history_stats(g.user["id"])
    return render_template("index.html", user=g.user, stats=stats, is_admin=g.is_admin)


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = (request.form.get("full_name") or "").strip()
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""
        redirect_target = get_safe_redirect_target()

        if len(full_name) < 2:
            flash("Please enter your full name.", "error")
        elif not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
        elif len(password) < 8:
            flash("Password must be at least 8 characters long.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif fetch_user_by_email(email):
            flash("An account with that email already exists.", "error")
        else:
            try:
                user_id = create_user(full_name, email, password)
            except sqlite3.IntegrityError:
                flash("An account with that email already exists.", "error")
            else:
                session.clear()
                session["user_id"] = user_id
                flash("Account created successfully. You can start verifying news now.", "success")
                return redirect(redirect_target)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = normalize_email(request.form.get("email"))
        password = request.form.get("password") or ""
        redirect_target = get_safe_redirect_target()
        user = fetch_user_by_email(email)

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            flash("Welcome back.", "success")
            return redirect(redirect_target)

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
@login_required()
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


@app.route("/feedback", methods=["POST"])
@login_required()
def submit_feedback():
    subject = (request.form.get("subject") or "").strip()
    message = (request.form.get("message") or "").strip()

    if len(subject) < 3:
        flash("Please enter a short feedback subject.", "error")
    elif len(message) < 10:
        flash("Please enter a more detailed feedback message.", "error")
    elif len(subject) > 120:
        flash("Feedback subject must stay under 120 characters.", "error")
    elif len(message) > 3000:
        flash("Feedback message must stay under 3000 characters.", "error")
    else:
        save_feedback(g.user, subject, message)
        flash("Thank you. Your feedback has been submitted.", "success")

    return redirect(url_for("dashboard"))


@app.route("/admin/feedback", methods=["GET"])
@admin_required
def admin_feedback():
    feedback_items = fetch_feedback_items()
    return render_template(
        "admin_feedback.html",
        user=g.user,
        is_admin=g.is_admin,
        feedback_items=feedback_items,
    )


@app.route("/predict", methods=["POST"])
@login_required(api=True)
def predict():
    global model

    if model is None:
        model = load_model_objects()

    if model is None:
        return (
            jsonify(
                {
                    "error": "Comparison model not found. Please run 'python train_model.py' first."
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

    web_validation = validate_news_online(news_text)
    verification = build_comparison_verification(web_validation)

    save_prediction(
        g.user["id"],
        news_text,
        cleaned_text,
        verification["final_verdict"],
        verification["match_score"],
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
    status = "ok" if model is not None else "model_not_ready"
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
