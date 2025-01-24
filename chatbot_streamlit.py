from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from huggingface_hub import InferenceApi
from API_key import Api_token
import streamlit as st

# llm
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = InferenceApi(repo_id=hf_model, token=Api_token)

# embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "content/"

embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# load Vector Database
vector_db = FAISS.load_local("content/faiss_index_chatbot", embeddings, allow_dangerous_deserialization=True)

# retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# prompt
template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {input}
Response:"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# bot with memory
@st.cache_resource
def init_bot():
    document_chain = StuffDocumentsChain(llm, prompt)
    return RetrievalQA(llm=llm, retriever=retriever, combine_documents_chain=document_chain)

rag_bot = init_bot()

##### streamlit #####
st.title("A Pioneer of Food Remedies")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_input := st.chat_input("Curious minds wanted!"):

    # Display user message in chat message container
    st.chat_message("human").markdown(user_input)

    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": user_input})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Going down the rabbithole for answers..."):

        # send question to chain to get answer
        answer = rag_bot({"query": user_input})

        # extract answer from response
        response = answer["result"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
