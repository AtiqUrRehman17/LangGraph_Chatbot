# import the dependen
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Literal,Annotated
from langchain_core.messages import HumanMessage,SystemMessage,BaseMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests 
import random

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
# Initialize the model using the API key explicitly
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key = api_key)

# tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool

def calculater(first_num:float,second_num:float,operation:str) -> dict:
    '''
    perform basic arithmetic operation on two numbers
    supported operations : add,sub,mul,div
    '''
    try :
        if operation == 'add':
            result = first_num + second_num
        elif operation == 'sub':
            result = first_num - second_num
        elif operation == 'mul':
            result = first_num * second_num
        elif operation == 'div':
            if second_num == 0:
                return {'erro':'Division by zero is not allowed'}
                result = first_num / second_num
            else:
                return {'error':f'Unsupported operation {operation}'}
        return {'first_num':first_num,'second_num':second_num,'operation':operation,'result':result}
    except Exception as e:
        return {'error',str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()
# make a list of all tools
tools = [search_tool,calculater,get_stock_price]

# bind the tools to the llm
llm_with_tool = llm.bind_tools(tools)



class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]
    

def chat_node(state:ChatState):
    '''
    LLM node that may answer or request a tool call
    '''
    messages = state['messages']
    result = llm_with_tool.invoke(messages)
    return {'messages':[result]}

tool_node = ToolNode(tools) # execute tool call


conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# define checkpointer
checkpointer = SqliteSaver(conn)


# graph building
graph = StateGraph(ChatState)
# adding nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')
chatbot = graph.compile()

chatbot = graph.compile(checkpointer=checkpointer)

# Helper function

def retrieve_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    return list(all_thread)
