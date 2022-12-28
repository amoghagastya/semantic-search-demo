import streamlit as st

st.set_page_config(
    page_title="Limitations", page_icon="ℹ", layout = "wide")



# st.sidebar.header("Knowledge Base")
st.markdown("# Current Bot Limitations")

st.write("Currently, the Amazon Shopping Assistant can only accurately converse about one topic at a time.")
st.write('''This is due to how the similar contexts are retrieved. The entire Chat transcript is provided as the input
         to query the Embedded Pinecone Database. This is done so that the entire context of the conversation can be preserved and queried.
         However, this comes at the cost of being able to talk about one topic at a time, needing a chat reset before a different
         topic can be answered accurately.''')

st.write("Please do let me know if you find a solution to this challenge!")
