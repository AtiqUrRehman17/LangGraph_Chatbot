# import the dependen
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Literal,Annotated
from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Initialize the model using the API key explicitly
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = api_key)

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    

def chat_node(state: ChatState):

    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    return {'messages': [response]}


conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)





# define checkpointer
checkpointer = SqliteSaver(conn)
# define a graph
graph = StateGraph(ChatState)
# add node
graph.add_node('chat_node',chat_node)
# add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    
    return (list(all_thread))