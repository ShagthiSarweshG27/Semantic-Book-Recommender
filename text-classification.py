import pandas as pd
books = pd.read_csv("books_cleaned.csv")

books["categories"].value_counts().reset_index()

books["categories"].value_counts().reset_index().query("count > 50")

books[books["categories"] == "Juvenile Fiction"]

category_mapping = {'Fiction': "Fiction",
'Juvenile Fiction': "Children's Fiction",
'Biography & Autobiography': "Nonfiction",
'History': "Nonfiction",
'Literary Criticism': "Nonfiction",
'Philosophy': "Nonfiction",
'Religion': "Nonfiction",
'Comics & Graphic Novels': "Fiction",
'Drama': "Fiction",
'Juvenile Nonfiction': "Children's Nonfiction",
'Science': "Nonfiction",
'Poetry': "Fiction"}
books["simple_categories"] = books["categories"].map(category_mapping)

books[~(books["simple_categories"].isna())]

from transformers import pipeline
fiction_categories = ["Fiction", "Nonfiction"]
pipe = pipeline("zero-shot-classification",
                model="facebook/bart-large-mnli")

sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[0]

pipe(sequence, fiction_categories)

import numpy as np
max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
max_label = pipe(sequence, fiction_categories)["labels"][max_index]
print(max_label)

def generate_predictions(sequence, categories):
    predictions = pipe(sequence, categories)
    max_index = np.argmax(predictions["scores"])
    max_label = predictions["labels"][max_index]
    return max_label

from tqdm import tqdm
actual_cats = []
predicted_cats = []
for i in tqdm(range(0,300)):
    sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Fiction"]
for i in tqdm(range(0,300)):
    sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    actual_cats += ["Nonfiction"]
    
predictions_df = pd.DataFrame({"actual_categories": actual_cats, "predicted_categories": predicted_cats})
predictions_df.head()

predictions_df["correct_prediction"] = (
    np.where(predictions_df["actual_categories"] == predictions_df["predicted_categories"], 1, 0)
)

predictions_df["correct_prediction"].sum() / len(predictions_df)

isbns = []
predicted_cats = []
missing_cats = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)

for i in tqdm(range(0, len(missing_cats))):
    sequence = missing_cats["description"][i]
    predicted_cats += [generate_predictions(sequence, fiction_categories)]
    isbns += [missing_cats["isbn13"][i]]
    
predicted_cats = predicted_cats[:len(isbns)]

missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_cats})
    
books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"])
books = books.drop(columns= ["predicted_categories"])

books[books["categories"].str.lower().isin([
    "romance",
    "science fiction",
    "scifi",
    "fantasy",
    "horror",
    "mystery",
    "thriller",
    "comedy",
    "crime",
    "historical"
])]

books.to_csv("books_with_categories.csv", index=False)