from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
import pandas as pd
import re
from pinecone import Pinecone

openai.api_key = "sk-7gZUem5hac1JNfM5BAPj4C-lWrnMaU68g4sq37yW97T3BlbkFJ2hJSTAkbhVBAfq5-uZiupdGtJrqtjfwfR4BALMKykA"

pc = Pinecone(api_key="6bf7c103-f831-41c6-80a1-27bd305c0acb")
index = pc.Index("ecomchatbot")
model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

def clean_data(df):
    # Remove rows with missing values in key columns
    df = df.dropna(subset=['name', 'description', 'p_attributes'])
    # Clean textual columns
    df['description'] = df['description'].apply(clean_text)
    df['p_attributes'] = df['p_attributes'].apply(clean_text)
    # Remove duplicates
    df = df.drop_duplicates(subset=['name', 'description', 'p_attributes'])
    # Ensure consistency
    df['colour'] = df['colour'].str.lower()
    df['brand'] = df['brand'].str.lower()
    return df

def create_embeddings_and_index(df):
    df['combined_features'] = df['name'] + " " + df['description'] + " " + df['p_attributes']
    df['embeddings'] = df['combined_features'].apply(lambda x: model.encode(x).tolist())
    # Indexing in Pinecone
    for i, row in df.iterrows():
        index.upsert(vectors=[(str(row['p_id']), row['embeddings'], {'text': row['combined_features'], 'name': row['name'], 'price': row['price'], 'img': row['img']})])

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string


# Load your dataset
df = pd.read_csv('fashion_dataset.csv')

# Clean the data
df = clean_data(df)

# Create embeddings and index in Pinecone
create_embeddings_and_index(df)