# A simple reactive agent that uses the ChatGroq model to process user requests and utilize tools for searching news, getting weather information, and translating text. The agent is built using the langchain library and is designed to be easily extendable with additional tools and functionalities. The agent's state is managed using a state graph, allowing for a structured flow of operations.

# Imports
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from utils.language_code import LANGUAGE_CODES
from utils.helper_functions import search_duckduckgo_news, get_weather, translate_text
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os
from dotenv import load_dotenv
load_dotenv()  

# Define the tool for searching DuckDuckGo
@tool("search_duckduckgo_news", description="Searches DuckDuckGo for news articles related to the query. Returns a list of search results with titles and snippets.")
def search_duckduckgo_tool(query: str) -> str:
    return search_duckduckgo_news(query)

# Define the tool for getting weather information
@tool("get_weather", description="Gets current weather for a city using Open-Meteo.")
def get_weather_tool(city: str) -> str:
    return get_weather(city)

# Define the tool for translating text
@tool("translate_text", description=f"Translates text to a target language using Google Translator. Supported languages: {LANGUAGE_CODES}")
def translate_text_tool(text: str, target_lang: str = "en") -> str:
    return translate_text(text, target_lang)

# Initialize the LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))

# Create the agent with the defined tool and system prompt
tools = [search_duckduckgo_tool, get_weather_tool, translate_text_tool]
agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant that can process user requests and fullfill them using the available tools.")

# Define the state and the function to run the agent
class AgentState(TypedDict):
    user_input: str
    llm_response: str

# Function to run the agent with the given state
def run_agent(state: AgentState) -> AgentState:
    
    result = agent.invoke({"messages": [{"role": "user", "content": state["user_input"]}]})
    state["llm_response"] = result["messages"][-1].content

    return state

# Build the state graph for the agent
builder = StateGraph(AgentState)

builder.add_node("agent", run_agent)
builder.set_entry_point("agent")
builder.add_edge("agent", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state: AgentState = {
        "user_input": "Search for news about Bihar and Translate to Hindi.",
        "llm_response": ""
    }

    final_state = graph.invoke(initial_state)
    print("LLM Response:")
    print(final_state["llm_response"])
