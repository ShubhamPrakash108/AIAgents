# Imports
import operator
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from utils.language_code import LANGUAGE_CODES
from typing import List, Tuple, Annotated, Literal
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

"""
groq.BadRequestError: Error code: 400 - {'error': {'message': "'response_format' : one of the following must be satisfied[('response_format' : value must be an object) OR ('response_format' : value must be an object) OR ('response_format' : value must be an object)]", 'type': 'invalid_request_error'}}
During task with name 'plan_check' and id 'c16f6854-b221-d7a4-f360-732e2580f558'
"""

# class YesNoResponse(BaseModel):
#     answer: Literal["YES", "NO"]

# llm_with_yes_no = ChatGroq(model="openai/gpt-oss-20b", temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))
# yes_no_llm = llm_with_yes_no.with_structured_output(YesNoResponse, response_format="json")

# State
class PlannerAgentExecuter(TypedDict):
    user_input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    llm_response: str
    plan_check_reflection: str 
    step_execution_reflection: str

def ask_yes_no(prompt: str) -> str:
    full_prompt = prompt + '\n\nRespond ONLY with a valid JSON object, no markdown: {"answer": "YES"} or {"answer": "NO"}'
    for _ in range(3):
        response = llm.invoke(full_prompt)
        try:
            text = response.content.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            if parsed.get("answer") in ("YES", "NO"):
                return parsed["answer"]
        except (json.JSONDecodeError, KeyError):
            continue
    return "YES"

# Nodes
def planner_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    prompt = f"""You are a planner. Break the following task into a list of simple executable steps.

Task: {state['user_input']}

Respond ONLY with a valid JSON object in this exact format, no markdown, no explanation:
{{"steps": ["step 1", "step 2", "step 3"]}}"""

    response = llm.invoke(prompt)
    
    # Strip markdown fences if model adds them
    text = response.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group())
    return {"plan": parsed["steps"]}

# Reflection node to check if the steps are correct or not
def reflection_planning_check_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    answer = ask_yes_no(f"""You are a planner checker. Does this plan correctly and completely address the original task?
    Original task: {state['user_input']}
    Plan: {state['plan']}""")
    return {"plan_check_reflection": answer}

# def reflection_planning_check_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
#     prompt = f"""
#     You are a planner checker. Check if the following plan correctly and completely addresses the original task. 
    
#     Original task: {state['user_input']}
#     Plan: {state['plan']}

#     Respond ONLY with a JSON object like: {{"answer": "YES"}} or {{"answer": "NO"}}
#     """
#     response = yes_no_llm.invoke(prompt)
#     if response.answer not in ("YES", "NO"):
#         raise ValueError(f"Invalid response from yes_no_llm: {response.answer}")
#     return {**state, "plan_check_reflection": response.answer}


def execute_step(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    if not state["plan"]:
        return state
    task = state["plan"][0]
    result = agent.invoke({"messages": [("user", task)]})
    remaining_plan = state["plan"][1:]
    return { "plan": remaining_plan,"past_steps": [(task, result["messages"][-1].content)]}


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


def check_correctness_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
    answer = ask_yes_no(f"""You are a task completion checker. Is the task fully and correctly completed?
    Original task: {state['user_input']}
    Steps completed: {state['past_steps']}
    Final response: {state.get("llm_response")}""")
    return { "step_execution_reflection": answer}

# def check_correctness_node(state: PlannerAgentExecuter) -> PlannerAgentExecuter:
#     prompt = f"""
#     You are a task completion checker. Is the following task fully and correctly completed?

#     Original task: {state['user_input']}
#     Steps completed: {state['past_steps']}
#     Final response: {state['llm_response']}

#     Respond ONLY with a JSON object like: {{"answer": "YES"}} or {{"answer": "NO"}}
#     """
#     response = yes_no_llm.invoke(prompt)
#     return {**state, "step_execution_reflection": response.answer}

def route_after_reflection(state):
    return END if state["step_execution_reflection"] == "YES" else "planner"

# Graph
builder = StateGraph(PlannerAgentExecuter)

builder.add_node("planner", planner_node)
builder.add_node("plan_check", reflection_planning_check_node)
builder.add_node("executor", execute_step)
builder.add_node("replan", replan_step)
builder.add_node("check", check_correctness_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "plan_check")
builder.add_conditional_edges(
    "plan_check",
    lambda state: "executor" if state["plan_check_reflection"] == "YES" else "planner",
    {"executor": "executor", "planner": "planner"}
)
builder.add_edge("executor", "replan")
builder.add_conditional_edges(
    "replan",
    lambda state: "check" if state.get("llm_response") else "executor",
    {"check": "check", "executor": "executor"}
)
builder.add_conditional_edges(
    "check",
    route_after_reflection,
    {"planner": "planner", END: END}
)

graph = builder.compile()

image_data = graph.get_graph().draw_mermaid_png()
with open("planning_and_reflection_agent.png", "wb") as f:
    f.write(image_data)

response = graph.invoke({
    "user_input": "Search the web and answer based only on the search results: Which city was called Patliputra in ancient times and what is its current weather of that city? Reply in Hindi in short",
    "plan_check_reflection": "",
    "step_execution_reflection": "",
    "llm_response": "",
    "plan": [],
    "past_steps": [],
})
print(response)