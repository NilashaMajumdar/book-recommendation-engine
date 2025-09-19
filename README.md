# 📚 Book Recommendation & Analysis with LLMs

_Looking for your next great read?_

This **end-to-end NLP** project helps you discover books in a way that goes beyond simple keywords. Using the power of
**large language models (LLMs)**,
semantic search, and sentiment analysis, the app recommends titles from a dataset of over **7,000 books** that truly
match
your interests, genres, and even your mood✨
---

## ✨ What You Can Do

With this app, you can:

- **Input your query** and get book recommendations through semantic vector search.
- **Filter results by genre**, narrowing down to fiction, non-fiction, or children’s categories.
- **Explore books by emotional tone**, discovering titles that match your mood.
- **Use the Gradio dashboard** to interact with the system in real time.

---

## 🛠️ How It Works

1. **Data Preparation**: Cleaned a dataset of **7,000 book descriptions**, removing incomplete entries, short texts, and
   inconsistent categories.
2. **Semantic Vector Search**: Generated vector embeddings with **OpenAI** and **LangChain** and built a **vector
   database** for
   similarity-based book retrieval.
3. **Genre Classification**: Used **BART (Hugging Face)** for **one-shot classification**, mapping **~500** original
   categories into **fiction, non-fiction, children’s fiction, and children’s non-fiction** with **0.75 F1-score**.
4. **Sentiment & Emotion Analysis**: Applied a fine-tuned **DistilBERT** model to detect tones like joy, sadness, fear,
   anger, and suspense .
5. **Interactive Dashboard**: Built with **Gradio** for real-time search, filtering, and exploration.

---

## 🚀 Tech Stack

- **Python** (Pandas, NumPy, Scikit-learn)
- **OpenAI embeddings** for semantic vector search
- **LangChain** for text splitting and integration
- **BART (Hugging Face)** for one-shot genre classification
- **DistilBERT (fine-tuned)** for sentiment & emotion analysis
- **Gradio** for the interactive dashboard
- **Chroma** for Vector Database

---

## 📊 Results

- Achieved **0.75 F1-score** on genre classification.
- DistilBERT-based sentiment analysis captured five distinct tones (joy, sadness, fear, anger, suspense) with high
  accuracy on validation samples.
- Built a vector search engine returning semantically similar book matches in under a few seconds.

---

## 🔮 Future Improvements

- Expand genre classification beyond 4 groups for finer recommendations.
- Add user history + authentication for personalized recommendations
- Scale pipeline to larger datasets (100k+ books) with distributed vector search.
- Build a simple frontend (React/Vue) connected to the FastAPI backend
