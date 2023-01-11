import streamlit as st
import pinecone
import openai
# from openai.embeddings_utils import get_embedding
import json
import os
# from streamlit_chat import message as st_message
import streamlit.components.v1 as components  # Import Streamlit
from streamlit_chat import message


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
# search = st.container()

# query = st.text_input('Ask a product related question!', key="input_text", on_change=generate_answer)

# components.html('''<div id="chat" align=center>
# <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
# <df-messenger
#   intent="WELCOME"
#   chat-title="AmazonAssistant"
#   agent-id="7aa00f7c-01c0-4afd-b1e3-d88026d64389"
#   language-code="en"
# ></df-messenger></div>''',height=510, width = 550)


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
    
    **Note**: Hit reset chat before asking it a question of a different topic. Check the limitations tab for the current challenges.
    
    The app is powered using OpenAI's embedding service with Pinecone's vector database. The whole process consists
    of *three* steps:
    
    **1**. User Queries are fed into OpenAI's embeddings service to generate a {'2048'}-dimensional query vector.
    
    **2**. We use Pinecone to identify similar context vectors (previously encoded from the Amazon Dataset).

    **3**. Relevant contexts are passed in a new question to OpenAI's generative model, returning our answer.

    **How do I make something like this?**

    It's easy! Book a [free Discovery Call with me](https://calendly.com/amagastya/20min) and I'll get your bot setup in no-time!
""")

if 'bot' not in st.session_state:
    st.session_state['bot'] = ["👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]

if 'user' not in st.session_state:
    st.session_state['user'] = ["Hi"]

if 'convo' not in st.session_state:
    st.session_state['convo'] = ["AI: 👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]

INDEX = 'amzn-semantic-search'

def load_index():
    pinecone.init(
        api_key=st.secrets["PINECONE_KEY"],  # app.pinecone.io
        environment='us-west1-gcp'
    )
    index_name = 'amzn-semantic-search'
    if not index_name in pinecone.list_indexes():
        raise KeyError(f"Index '{index_name}' does not exist.")

    return pinecone.Index(index_name)

# index = load_index()

@st.experimental_singleton(show_spinner=False)
def init_key_value():
    with open('amzn-mapping.json', 'r') as fp:
        mappings = json.load(fp) 
    return mappings

with open('amzn-mapping.json', 'r') as fp:
        mappings = json.load(fp) 

openai.api_key = st.secrets["OPENAI_KEY"]
    
def get_embedding(text, engine="text-similarity-babbage-001"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=engine)['data'][0]['embedding']

def create_context(question, index, mappings, max_len=3750, size="babbage"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-search-{size}-query-001')
    res = index.query(q_embed, top_k=2, include_metadata=True)
    
    cur_len = 0
    contexts = []

    for row in res['matches']:
        text = mappings[row['id']]
        cur_len += row['metadata']['n_tokens'] + 4
        if cur_len < max_len:
            contexts.append(text)
        else:
            cur_len -= row['metadata']['n_tokens'] + 4
            if max_len - cur_len < 200:
                break
    return "\n\n###\n\n".join(contexts)

chat = "I am a helpful AI Amazon Shopping assistant created by Amogh. The following is a chat conversation between me and a user."
instructions = {
    "Conservative Q&A": "I am a helpful AI Amazon Shopping assistant created by Amogh. The following is a chat conversation between me and a user. Write a paragraph, addressing the user's question, and use the context below to obtain relevant information. The default currency is Rupees (₹). If the question absolutely cannot be answered based on the context below, say I dont know. Refuse to answer anything negative or abusive.\"\n\nContext:\n{0}\n\n---\n\nChat: {1} ",
    "Chit-chat Allowed" : "I am a helpful AI assistant created by Amogh. The following is a chat conversation between me and a user. Answer the user's question and use the context below for additional info. Allow for chit-chat and general conversation with the user, but refuse to answer anything negative or abusive.\"\n\nContext:\n{0}\n\n---\n\nChat: {1} ",
    "Answer in 1-2 lines" : chat + "Use the context below for relevant information and answer the user's question in 1-2 lines at maximum.\nContext:\n{0}\n\n---\n\nChat: {1} ",
    "Summarize the Conversation" : chat + "Write a paragraph long summary about the user's query given the relevant context below. \"\n\nContext:\n{0}\n\n---\n\nChat: {1} ",
}

convo = st.session_state['convo']

def chat(
    index,
    fine_tuned_qa_model="text-davinci-003",
    question="i need a sun screen product",
    instruction="Answer the query based on the context below, and if the query can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:",
    max_len=3550,
    size="babbage",
    debug=False,
    max_tokens=400,
    stop_sequence=None,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    convo.append("User: " + question)
    # remove convo newlines and concat to ctx
    ctx =  ('').join(("\n").join(convo).splitlines())
    context = create_context(
        ctx,
        index,
        mappings,
        max_len=max_len,
        size=size,
    )
    if debug:
        print("Context:\n" + context)
        print("\n\n")
    try:
        # fine-tuned models requires model parameter, whereas other models require engine parameter
        model_param = (
            {"model": fine_tuned_qa_model}
            if ":" in fine_tuned_qa_model
            and fine_tuned_qa_model.split(":")[1].startswith("ft")
            else {"engine": fine_tuned_qa_model}
        )
        print('convo so far ', convo)
        print(instruction.format(context, question))
        response = openai.Completion.create(
            prompt=instruction.format(context, ("\n").join(convo)),
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            **model_param,
        )
        convo.append(response["choices"][0]["text"].strip())
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

def generate_ans(user_input, style):
    print('generating answer...')
    st.session_state.user.append(user_input)
    st.session_state.convo.append("User: " + user_input)
    ctx = ('\n').join(st.session_state.convo)
    print('convo so far', st.session_state.convo)
    st.session_state.convo.append("AI: ")
    result = chat(index, question=user_input, 
                            instruction = instructions[style], debug=False) 
    return st.session_state.bot.append(result)

# user_input = get_text()
with st.spinner("Connecting to OpenAI..."):
    openai.api_key = st.secrets["OPENAI_KEY"]

with st.spinner("Connecting to Pinecone..."):
    index = load_index()
    text_map = init_key_value()

def clear_text():
    st.session_state["text"] = ""
        
def main():
    search = st.container()
    query = search.text_input('Ask a product related question!', value="", key="text")
    
    with search.expander("Chat Options"):
        style = st.radio(label='Style', options=[
            'Conservative Q&A',
            'Chit-chat Allowed'
        ],on_change=clear_text)  
    
    # search.button("Go!", key = 'go')
    if search.button("Go!") or query != "":
        with st.spinner("Retrieving, please wait..."):
            # lowercase relevant lib filters
            # ask the question
            answer = generate_ans(query, style=style)
            # clear_text()            
            # return 
    if st.button("Reset Chat", on_click=clear_text):
        st.session_state['bot'] = ["👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]
        st.session_state['user'] = ["Hi"]
        st.session_state['convo'] = ["AI: 👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]
        
    if st.session_state['bot']:
            for i in range(len(st.session_state['bot'])-1, -1, -1):
                message(st.session_state["bot"][i], key=str(i))
                message(st.session_state['user'][i], is_user=True, key=str(i) + '_user')

main()
