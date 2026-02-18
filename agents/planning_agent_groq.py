# Imports
import operator
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from utils.language_code import LANGUAGE_CODES
from typing import List, Tuple, Annotated
from utils.helper_functions import search_duckduckgo_news, get_weather, translate_text, web_search_duckduckgo
from langgraph.graph import StateGraph, END
from typing import TypedDict
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Tools
@tool("search_duckduckgo_news", description="Searches DuckDuckGo for news articles related to the query.")
def search_duckduckgo_tool(query: str) -> str:
    return search_duckduckgo_news(query)

@tool("get_weather", description="Gets current weather for a city using Open-Meteo.")
def get_weather_tool(city: str) -> str:
    return get_weather(city)

@tool("translate_text", description=f"Translates text to a target language. Supported languages: {LANGUAGE_CODES}")
def translate_text_tool(text: str, target_lang: str = "en") -> str:
    return translate_text(text, target_lang)

@tool("web_search_duckduckgo", description="Searches DuckDuckGo for general web results related to the query.")
def web_search_duckduckgo_tool(query: str) -> str:
    return web_search_duckduckgo(query)

# LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))

# Agent
tools = [search_duckduckgo_tool, get_weather_tool, translate_text_tool, web_search_duckduckgo_tool]
agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant. Use the available tools to complete tasks.")


# State
class PlannerAgentExecuter(TypedDict):
    user_input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    llm_response: str

# Nodes
def planner_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    prompt = f"""You are a planner. Break the following task into a list of simple executable steps.

Task: {state['user_input']}

Respond ONLY with a valid JSON object in this exact format, no markdown, no explanation:
{{"steps": ["step 1", "step 2", "step 3"]}}"""

    response = llm.invoke(prompt)
    
    # Strip markdown fences if model adds them
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)
    return {"plan": parsed["steps"]}

def execute_step(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    task = state["plan"][0]
    result = agent.invoke({"messages": [("user", task)]})
    return {"past_steps": [(task, result["messages"][-1].content)]}

def replan_step(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    prompt = f"""You are a replanner. 

Original task: {state['user_input']}
Completed steps so far: {state['past_steps']}

Is the task fully complete based on completed steps?

Respond ONLY with a valid JSON object, no markdown, no explanation:
If complete:   {{"is_done": true, "final_answer": "your final answer here", "remaining_steps": []}}
If incomplete: {{"is_done": false, "final_answer": "", "remaining_steps": ["next step 1", "next step 2"]}}"""

    response = llm.invoke(prompt)
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)

    if parsed["is_done"]:
        return {"llm_response": parsed["final_answer"]}
    return {"plan": parsed["remaining_steps"]}

def should_end(state: PlannerAgentExecuter):
    return END if state.get("llm_response") else "executor"

# Graph
builder = StateGraph(PlannerAgentExecuter)
builder.add_node("planner",  planner_node)
builder.add_node("executor", execute_step)
builder.add_node("replan",   replan_step)

builder.set_entry_point("planner")
builder.add_edge("planner",  "executor")
builder.add_edge("executor", "replan")
builder.add_conditional_edges("replan", should_end, ["executor", END])

graph = builder.compile()

response = graph.invoke({"user_input": "Search the web and answer based only on the search results: Which city was called Patliputra in ancient times and what is its current weather of that city?"})
print(response["llm_response"])
print(response["plan"])