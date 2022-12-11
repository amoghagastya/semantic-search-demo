import streamlit as st
import pinecone
import openai
# from openai.embeddings_utils import get_embedding
import json
import os
# from streamlit_chat import message as st_message
import streamlit.components.v1 as components  # Import Streamlit


st.set_page_config(
    page_title="Hello",
    page_icon="🤖",
    layout = "centered"
)


# st.markdown("""
# <link
#   rel="stylesheet"
#   href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap"
# />
# """, unsafe_allow_html=True)


st.subheader("Chat with your Amazon Shopping Assistant! 👇")

search = st.container()

# query = st.text_input('Ask a product related question!', key="input_text", on_change=generate_answer)

components.html('''<div id="chat" align=center>
<script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
<df-messenger
  intent="WELCOME"
  chat-title="AmazonAssistant"
  agent-id="7aa00f7c-01c0-4afd-b1e3-d88026d64389"
  language-code="en"
></df-messenger></div>''',height=500, width = 550)

# styl = f"""
# <style>
#     .chat {
#         width = 30%
#     }
# </style>
# """

st.sidebar.write(f"""
    ### How it Works

    The bot takes Product titles and Descriptions from the Amazon India Product 
    Dataset and collates their content into a natural language search and a Conversation tool.

    Ask product related questions like **"Suggest some christmas gift ideas"** or **"What should I buy for
    my 10 year old brother?"**, or **"I'm looking for organic products"** and it returns relevant results!
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. User Queries are fed into OpenAI's embeddings service to generate a {'4096'}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from the Amazon Dataset).

    **3**. Relevant contexts are passed in a new question to OpenAI's generative model, returning our answer.

    **How do I make something like this?**

    It's easy! Book a [free Discovery Call with me](https://calendly.com/amagastya/15min) and I'll get your bot setup in no-time!
""")
