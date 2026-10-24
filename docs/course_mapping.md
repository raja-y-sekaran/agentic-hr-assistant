# Where each course topic lives in this repo

Built while taking the ProITBridge *Advanced Agentic AI* course. The course projects used the course's own demo
handbook and code; this repository is a **rebuild in my own code** around a fictional company and handbook, so no
course files are reproduced. The topics, and where each one is implemented:

| Course module | Topic | In this repo |
|---|---|---|
| 2 | LLM APIs and AI communication | `src/core/settings.py`, model set in `.env` |
| 3 | Context window, conversation history, memory | checkpointer and `thread_id` (LangChain, LangGraph); session db (Agno) |
| 4 / 4.3 | RAG, and the top 8 chunking strategies | `src/features/chunking.py`, `src/models/retriever.py`, `reports/chunking_comparison.csv` |
| 5.1 / 5.3 | Tool calling; many tools vs one agent | `src/core/hr_tools.py`, `src/agents/toolkit.py` |
| 5.2 | JSON for agentic AI | the `HRAnswer` output schema and JSON tool results |
| 5.4 | Security layer between the LLM's response and the application | `src/security/guard.py` |
| 6.1–6.4 | LangChain, `create_agent`, the HR assistant, tool validation | `src/agents/langchain_agent.py` |
| 7.x | LangGraph foundations and project | `src/agents/langgraph_agent.py`, `langgraph.json` (LangGraph Studio) |
| 8.1 / 8.3 | Agno multi-agent teams, HR assistant | `src/agents/agno_team.py` |
| 8.2.x | CrewAI multi-agent crews, HR assistant | `src/agents/crewai_crew.py` |
| 10.1 | LangSmith tracing | set `LANGSMITH_TRACING=true` and a key in `.env`; LangChain and LangGraph runs are traced with no code change |
