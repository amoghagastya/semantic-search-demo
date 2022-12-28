import streamlit as st
from streamlit_chat import message
import requests
import os
import pinecone
import openai
import json
import time

st.set_page_config(
    page_title="Streamlit Chat - Demo",
    page_icon=":robot:"
)

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
headers = {"Authorization": os.environ.get('api_key')}

st.header("Streamlit Chat - Demo")
st.markdown("[Github](https://github.com/ai-yash/st-chat)")

if 'bot' not in st.session_state:
    st.session_state['bot'] = ["👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]

if 'user' not in st.session_state:
    st.session_state['user'] = ["Hi"]

if 'convo' not in st.session_state:
    st.session_state['convo'] = ["AI: 👋 Greetings! I'm your virtual shopping assistant 🛍️ How may I help you today?"]

# def query(payload): 
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()

INDEX = 'amzn-semantic-search'

def load_index():
    pinecone.init(
        api_key='d9b601c8-7d02-4c58-828e-d306cc6bc45a',  # app.pinecone.io
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
# @st.experimental_singleton(show_spinner=False)
# def init_openai():
#     # initialize connection to OpenAI
#     openai.api_key = st.secrets["OPENAI_KEY"]

openai.api_key = st.secrets["OPENAI_KEY"]
    
def get_embedding(text, engine="text-similarity-babbage-001"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=engine)['data'][0]['embedding']

def create_context(question, index, mappings, max_len=3750, size="babbage"):
    """
    Find most relevant context for a question via Pinecone search
    """
    q_embed = get_embedding(question, engine=f'text-search-{size}-query-001')
    res = index.query(q_embed, top_k=3, include_metadata=True)
    
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
    
# def get_text():
#     input_text = st.text_input("You: ", key="input", on_change=generate_ans)
#     return input_text 

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

# def get_ans():
#     # query = ""
#     return query
def clear_text():
    st.session_state["text"] = ""

# def clear_text(query, style):
#     with st.spinner("Retrieving, please wait..."):
#             # lowercase relevant lib filters
#             # ask the question
#         answer = generate_ans(query, style=style)
#         st.session_state["text"] = ""
        
def main():
    search = st.container()
    query = search.text_input('Chat with your Amazon Shopping Assistant! 👇', value="", key="text")
    
    with search.expander("Chat Options"):
        style = st.radio(label='Style', options=[
            'Conservative Q&A',
            'Chit-chat Allowed'
        ],on_change=clear_text)  
    
    # search.button("Go!", key = 'go')
    if search.button("Go!", on_click=clear_text) or query != "":
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
    
# import streamlit as st
    # text = st.empty()
    # text.text_input("input something", value="", key="1")
    # time.sleep(5)
    # text.text_input("input something", value="", key="2")
                
main()