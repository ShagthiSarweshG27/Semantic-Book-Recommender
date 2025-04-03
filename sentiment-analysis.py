import pandas as pd
books = pd.read_csv("books_with_categories.csv")

from transformers import pipeline
classifier = pipeline("text-classification", 
                      model="j-hartmann/emotion-english-distilroberta-base", 
                      top_k = None)
classifier("I love this")

classifier(books["description"][0])

sentences = books["description"][0].split(".")
prediction = classifier(sentences)

sorted(prediction[0], key=lambda x:x["label"])

import numpy as np
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

def calculate_max_emotion_scores(predicition):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for predictions in prediction:
        sorted_predictions = sorted(predictions, key=lambda x: x["label"])
        for index, label in enumerate(emotion_labels):
            per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}

for i in range(10):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])
        
from tqdm import tqdm
emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)
    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])
        
emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn

books = pd.merge(books, emotions_df, on="isbn13")

books.to_csv("books_with_emotions.csv", index=False)