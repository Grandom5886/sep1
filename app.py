from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *
from openai import OpenAI

st.subheader("Ecommerce Chatbot")

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

llm = client.chat.completions.create(
  model="meta/llama-3.1-8b-instruct",
  messages=[
        {"role": "user", "content": "Hello, How can I help you today?"}
    ],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=False
)

# openai.api_key = "sk-7gZUem5hac1JNfM5BAPj4C-lWrnMaU68g4sq37yW97T3BlbkFJ2hJSTAkbhVBAfq5-uZiupdGtJrqtjfwfR4BALMKykA"

# llm = openai.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": "How can I help you today?"}
#     ]
# )

# llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="sk-7gZUem5hac1JNfM5BAPj4C-lWrnMaU68g4sq37yW97T3BlbkFJ2hJSTAkbhVBAfq5-uZiupdGtJrqtjfwfR4BALMKykA")

# Function to get or create a session-specific chat history
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if 'chat_histories' not in st.session_state:
        st.session_state['chat_histories'] = {}
    if session_id not in st.session_state['chat_histories']:
        st.session_state['chat_histories'][session_id] = InMemoryChatMessageHistory()
    return st.session_state['chat_histories'][session_id]

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = "1"  # Example session ID

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

# Create the RunnableWithMessageHistory instance
chain = RunnableWithMessageHistory(
    runnable=llm,  # The LLM model used for generating responses
    get_session_history=lambda: get_session_history(st.session_state['session_id'])  # Function to retrieve chat history for the session
)

response_container = st.container()
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = chain.invoke({"input": f"Context:\n {context} \n\n Query:\n{query}"})
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
