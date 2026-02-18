from langchain_groq import ChatGroq
from langchain.tools import tool
import operator
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

llm_model = "openai/gpt-oss-20b"

# Tools
@tool("search_duckduckgo_news", description="Searches DuckDuckGo for news articles related to the query.")
def search_duckduckgo_tool(query: str) -> str:
    return search_duckduckgo_news(query)

@tool("get_weather", description="Gets current weather for a city using Open-Meteo. This tool will be used to get the current temperature of any city.")
def get_weather_tool(city: str) -> str:
    return get_weather(city)

@tool("translate_text", description=f"Translates text to a target language. Supported languages: {LANGUAGE_CODES}")
def translate_text_tool(text: str, target_lang: str = "en") -> str:
    return translate_text(text, target_lang)

@tool("web_search_duckduckgo", description="Searches DuckDuckGo for general web results related to the query.")
def web_search_duckduckgo_tool(query: str) -> str:
    return web_search_duckduckgo(query)

# LLM
llm = ChatGroq(model=llm_model, temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))

# Agent
tools = [search_duckduckgo_tool, get_weather_tool, translate_text_tool, web_search_duckduckgo_tool]
agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant. Use the available tools to complete tasks.")

class NeedPlanningResponse(BaseModel):
    need_planning: Literal[True, False] 

class CheckAnswerResponse(BaseModel):
    is_answer_correct: Literal[True, False]

class HybridClassificationAgentState(TypedDict):
    user_input: str
    llm_response: str
    plan: List[str] 
    past_steps: Annotated[List[Tuple], operator.add] 
    need_planning: bool 
    is_answer_correct: bool
    retry_count: int  


plan_classifier_llm = ChatGroq(model=llm_model, temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))
check_answer_node_llm = ChatGroq(model=llm_model, temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))

def need_planning_reflection(state: HybridClassificationAgentState) -> HybridClassificationAgentState:
    prompt = f"""Given the user input: "{state['user_input']}", determine if the task requires a plan or can be answered directly. If it requires a plan, set need_planning to True. Otherwise, set need_planning to False.
    DON'T RESPOND WITH ANY EXPLANATION, JUST RETURN THE BOOLEAN VALUE FOR need_planning."""

    plan_classifier_llm_structured = plan_classifier_llm.with_structured_output(NeedPlanningResponse)
    response = plan_classifier_llm_structured.invoke([("system", "You are a helpful assistant that determines if a task requires planning or can be answered directly."), ("user", prompt)])
    need_planning = response.need_planning
    return {"need_planning": need_planning, "retry_count": 0}  


def direct_answer_node(state: HybridClassificationAgentState) -> HybridClassificationAgentState:
    result = agent.invoke({"messages": [("user", state["user_input"])]})
    answer = result["messages"][-1].content
    return {"llm_response": answer}

def planner_node(state: HybridClassificationAgentState) -> HybridClassificationAgentState:
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

def execute_step(state: HybridClassificationAgentState) -> HybridClassificationAgentState:
    if not state["plan"]:
        return state
    
    task = state["plan"][0]
    
    # Build context from past steps so agent remembers previous results
    context = ""
    if state["past_steps"]:
        context = "Here is what has been done so far:\n"
        for prev_task, prev_answer in state["past_steps"]:
            context += f"- Task: {prev_task}\n  Result: {prev_answer}\n"
        context += "\nUsing the above context, now complete the following task:\n"
    
    full_task = context + task
    
    result = agent.invoke({"messages": [("user", full_task)]})
    remaining_plan = state["plan"][1:]
    answer = result["messages"][-1].content
    return {"llm_response": answer, "plan": remaining_plan, "past_steps": [(task, answer)]}

def check_answer_node(state: HybridClassificationAgentState) -> HybridClassificationAgentState:
    
    prompt = f"""
    Based on the original user input: "{state['user_input']}" and the response from the llm: "{state['llm_response']}", determine if the response correctly and fully answers the user's query. If it does, set is_answer_correct to True. Otherwise, set it to False.
    DON'T RESPOND WITH ANY EXPLANATION, JUST RETURN THE BOOLEAN VALUE FOR is_answer_correct
    """
    check_answer_llm = check_answer_node_llm.with_structured_output(CheckAnswerResponse)
    response = check_answer_llm.invoke([("system", "You are a helpful assistant that checks if the answer provided correctly and fully answers the user's query."), ("user", prompt)])
    is_answer_correct = response.is_answer_correct
    return {"is_answer_correct": is_answer_correct, "retry_count": state.get("retry_count", 0) + 1}  


MAX_RETRIES = 3  

def route_after_checking_answer(state: HybridClassificationAgentState) -> str:
    if state["is_answer_correct"] or state.get("retry_count", 0) >= MAX_RETRIES:  
        return END
    else:
        return "planning"
    
def route_after_execute_step(state: HybridClassificationAgentState) -> str:
    if state["plan"]:  
        return "execute_step"
    else: 
        return "check_answer"


builder = StateGraph(HybridClassificationAgentState)

builder.add_node("need_planning_check", need_planning_reflection)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("planning", planner_node)
builder.add_node("execute_step", execute_step)
builder.add_node("check_answer", check_answer_node)

builder.set_entry_point("need_planning_check")

builder.add_conditional_edges(
    "need_planning_check",
    lambda state: "planning" if state["need_planning"] else "direct_answer",
    {"planning": "planning", "direct_answer": "direct_answer"}
)

builder.add_edge("direct_answer", "check_answer")
builder.add_edge("planning", "execute_step")
builder.add_conditional_edges(
    "execute_step",
    route_after_execute_step,
    {"execute_step": "execute_step", "check_answer": "check_answer"}
)

builder.add_conditional_edges(
    "check_answer",
    route_after_checking_answer,
    {"planning": "planning", END: END}
)

graph = builder.compile()

image_data = graph.get_graph().draw_mermaid_png()
with open("hybrid_classification_agent.png", "wb") as f:
    f.write(image_data)

result = graph.invoke({
    "user_input": "What is the capital of Bihar and current temperature of that city?",
    "llm_response": "",
    "plan": [],
    "past_steps": [],
    "need_planning": False,
    "is_answer_correct": False,
    "retry_count": 0
})

print("result:", result)