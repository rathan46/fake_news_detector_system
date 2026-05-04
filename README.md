# AI-Based Fake News Detection System

This is a beginner-friendly Python Flask project that predicts whether a piece of news text is **REAL** or **FAKE** using a stronger and more reliable machine learning model.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- scikit-learn
- pandas
- nltk
- SQLite

## Project Structure

```text
AI-Based Fake News Detection System/
|-- app.py
|-- .env
|-- .env.example
|-- train_model.py
|-- model.pkl
|-- vectorizer.pkl
|-- database.db
|-- requirements.txt
|-- README.md
|-- data/
|   |-- fake_or_real_news.csv
|-- templates/
|   |-- index.html
|-- static/
|   |-- style.css
|   |-- script.js
```

## Features

- Enter news text on the home page
- Click the **Verify News** button
- Get one combined final result using AI analysis and independent web checking
- View confidence percentage
- Save every prediction in SQLite history
- Show simple prediction statistics
- Cross-check news online using related internet articles and fetched article content
- Responsive clean UI
- Runs on localhost

## Setup Instructions

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the machine learning model:

```bash
python train_model.py
```

3. Start the Flask application:

```bash
python app.py
```

4. Open the browser and visit:

```text
http://127.0.0.1:5000/
```

## Flask Routes

- `GET /` - Show homepage
- `POST /predict` - Verify text using the AI model and related web coverage
- `GET /history` - Return saved prediction history
- `GET /health` - Return `OK`

## Notes

- A small sample dataset is included in `data/fake_or_real_news.csv` so the project can run locally right away.
- You can replace the sample dataset with a larger real-world dataset that uses the same columns: `text` and `label`.
- The database file `database.db` is created automatically when the app runs.
- The training script now uses better text cleaning, unigram + bigram TF-IDF features, and a balanced Logistic Regression model.
- If NLTK stopwords are unavailable on first run, the app falls back to a built-in stopword list.
- For internet validation, create a free API key from NewsAPI and add it to `.env` before starting the Flask app.
- The final verdict is a stronger combined heuristic, but no automated system can guarantee perfect truth verification for every live news story.

## Environment File

Create a file named `.env` in the project root and add:

```env
NEWS_API_KEY=your_newsapi_key_here
```

You can copy the included `.env.example` file and replace the placeholder value with your real key.

## Example Labels

- `REAL`
- `FAKE`

