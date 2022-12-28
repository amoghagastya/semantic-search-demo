import streamlit as st

st.set_page_config(
    page_title="Limitations", page_icon="ℹ", layout = "wide")



# st.sidebar.header("Knowledge Base")
st.markdown("# Current Bot Limitations")

st.write("Currently, the Amazon Shopping Assistant can only accurately converse about one topic at a time.")
st.write('''This is due to how the similar contexts are retrieved. The entire Chat transcript is provided as the input
         to query the Embedded Pinecone Database. This is done so that the context of the conversation can be preserved and queried.
         However, this comes at the cost of being able to talk about one topic at a time, needing a chat reset before a different
         topic can be answered accurately.''')

st.write("Please do let me know if you find a solution to this challenge!")


st.sidebar.write(f"""
    ### How it Works

    The bot takes Product titles and Descriptions from the Amazon India Product 
    Dataset and collates their content into a natural language search and a Conversation tool.

    Ask product related questions like **"Suggest some christmas gift ideas"** or **"What should I buy for
    my 10 year old brother?"**, or **"I'm looking for organic products"** and it returns relevant results!
    
    **Note**: Hit reset chat before asking it a question of a different topic. Check the limitations tab for the current challenges.
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. User Queries are fed into OpenAI's embeddings service to generate a {'2048'}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from the Amazon Dataset).

    **3**. Relevant contexts are passed in a new question to OpenAI's generative model, returning our answer.

    **How do I make something like this?**

    It's easy! Book a [free Discovery Call with me](https://calendly.com/amagastya/20min) and I'll get your bot setup in no-time!
""")
