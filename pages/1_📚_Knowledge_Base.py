import streamlit as st
import pandas as pd
# from streamlit_chat import message as st_message
import streamlit.components.v1 as components  # Import Streamlit

# st.set_page_config(
#     page_title="Knowledge Base",
#     page_icon="🤖",
#     layout = "centered"
# )

st.set_page_config(
    page_title="Knowledge Base", page_icon="ℹ", layout = "wide")

st.sidebar.write(f"""
    ### How it Works

    The bot takes Product titles and Descriptions from the Amazon India Product 
    Dataset and collates their content into a natural language search and a Conversation tool.

    Ask product related questions like **"Suggest some christmas gift ideas"** or **"What should I buy for
    my 10 year old brother?"**, or **"I'm looking for organic products"** and it returns relevant results!
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. Questions are fed into OpenAI's embeddings service to generate a {'2048'}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from Q&A pages).

    **3**. Relevant contexts are passed in a new question to OpenAI's generative model, returning our answer.

    **How do I make something like this?**

    It's easy! Book a [free Discovery Call with me](https://calendly.com/amagastya/20min) and I'll get your bot setup in no-time!
""")

# st.sidebar.header("Knowledge Base")
st.markdown("# Semantic Search - Knowledge Base")

st.write("Using Existing Knowledge Bases, we can integrate Semantic Search into Conversational Agents with ease.")
st.write("Note: the Amazon Shopping agent is not trained on ANY intents or training phrases and only uses the default fallback intent. Try typing something in and the bot will query the Knowledge-base below and return a response based on the closest semantic results.")

df = pd.read_parquet('./amzn-embeddings-1.parquet')
# df = df.astype({'id':'string'})
st.write("Our Vector Search Knowledge Database looks something like this -")
st.dataframe(df)
