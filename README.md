# AI Agents

A collection of AI agent architectures built with **LangChain**, **LangGraph**, and **Groq** LLMs. Each agent demonstrates a different agentic pattern — from simple reactive tool use to multi-step planning with self-reflection.

## Agents

| Agent | File | Description |
|-------|------|-------------|
| **Reactive Agent** | `agents/reactive_agent_groq.py` | A straightforward agent that receives user input and responds using available tools in a single pass. |
| **Planning Agent** | `agents/planning_agent_groq.py` | Breaks tasks into a step-by-step plan, executes each step sequentially, and replans until the task is complete. |
| **Planning & Reflection Agent** | `agents/planning_and_reflection_agent.py` | Extends the planning agent with reflection checkpoints — validates the plan before execution and verifies the final answer for correctness. |
| **Hybrid Classification Agent** | `agents/hybrid_classification_agent.py` | Classifies whether a query needs planning or can be answered directly, routes accordingly, and includes answer verification with retry logic. |

## Tools

All agents share a common set of tools defined in `utils/helper_functions.py`:

- **DuckDuckGo News Search** — Search for recent news articles.
- **DuckDuckGo Web Search** — General web search.
- **Weather Lookup** — Get current weather for any city via Open-Meteo (no API key needed).
- **Text Translation** — Translate text between 30+ languages using Google Translator.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ShubhamPrakash108/AIAgents
   cd AIAgents
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run an agent**
   ```bash
   python -m agents.reactive_agent_groq
   ```

## Project Structure

```
├── agents/
│   ├── reactive_agent_groq.py
│   ├── planning_agent_groq.py
│   ├── planning_and_reflection_agent.py
│   └── hybrid_classification_agent.py
├── utils/
│   ├── helper_functions.py
│   └── language_code.py
├── requirements.txt
└── README.md
```

