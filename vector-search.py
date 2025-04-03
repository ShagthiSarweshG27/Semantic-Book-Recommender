from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

books = pd.read_csv("books_cleaned.csv")

books["tagged_description"].to_csv("tagged_description.txt",
                                   sep = "\n",
                                   index = False,
                                   header = False)

raw_documents = TextLoader("tagged_description.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0,chunk_overlap=0,separator="\n")
documents = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = FAISS.from_documents(documents, embeddings)

query = "A book to teach children about nature"
docs = db_books.similarity_search(query, k=10)

books[books["isbn13"] == int(docs[0].page_content.split()[0].strip())]

def retrieve_semantic_recommendations(
        query: str,
        top_k: int = 10,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query,k=50)
    books_list=[]
    for i in range(0,len(recs)):
        books_list += [int(recs[i].page_content.strip('"').split()[0])]
    return books[books["isbn13"].isin(books_list)].head(top_k)

retrieve_semantic_recommendations("A book to teach children about nature")