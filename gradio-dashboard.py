import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

load_dotenv()

# Set thumbnail conditions for each book
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
# Books without thumbnail covers
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Set up the vector database (as done previously)
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()  # reading tagged desc into textloader
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)  # splits text based on new linws
documents = text_splitter.split_documents(raw_documents)  # splits each description into a new document
db_books = Chroma.from_documents(documents,
                                 OpenAIEmbeddings())  # get desc embeddings and store in chroma vectore databases


# Apply category filtering and sorting based on emotional tone
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)  # vector search based on given query
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # For category drop down menu
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    # For tone drop down menu
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Cutting down the description length to adjust for less space
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(
            truncated_desc_split[:30]) + "..."  # keep first 30 characters and then end with ...

        # Formatting author names based on number of authors
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:  # if only 1 author
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    # Title
    gr.Markdown("# Semantic book recommender")

    # Row-wise
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Sort by category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Sort by emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations!! :)")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    # Requirements to click on button
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
