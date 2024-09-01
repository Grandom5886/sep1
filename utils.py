import streamlit as st
import re
import pinecone
from sentence_transformers import SentenceTransformer

def get_model():
    if not hasattr(st.session_state, 'embedding_model'):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return st.session_state.embedding_model

def init_pinecone_index():
    if not hasattr(st.session_state, 'pinecone_index'):
        pinecone.init(api_key='6bf7c103-f831-41c6-80a1-27bd305c0acb', environment='us-east-1-aws')
        st.session_state.pinecone_index = pinecone.Index('ecomchatbot')
    return st.session_state.pinecone_index

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
    model = get_model()
    index = init_pinecone_index()

    df['combined_features'] = df['name'] + " " + df['description'] + " " + df['p_attributes']
    df['embeddings'] = df['combined_features'].apply(lambda x: model.encode(x).tolist())
    
    # Indexing in Pinecone
    for i, row in df.iterrows():
        index.upsert(vectors=[(str(row['p_id']), row['embeddings'], {
            'text': row['combined_features'], 
            'name': row['name'], 
            'price': row['price'], 
            'img': row['img']
        })])

def find_match(input):
    model = get_model()
    index = init_pinecone_index()

    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation_string, query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Refine the user's query based on the conversation history."},
            {"role": "user", "content": f"Conversation: {conversation_string}\nQuery: {query}"}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i+1] + "\n"
    return conversation_string
