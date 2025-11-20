# Multi-Framework AI Agent System

<img width="1837" height="858" alt="image" src="https://github.com/user-attachments/assets/aa6e8489-3b60-4ee5-a0c8-948688aa68e3" />
<img width="1817" height="833" alt="image" src="https://github.com/user-attachments/assets/d0897453-c8ae-4237-9d74-76d79fbf6ab0" />
<img width="1830" height="837" alt="image" src="https://github.com/user-attachments/assets/c0cf7434-1360-4b6c-859c-35c34a6b709f" />
<img width="1814" height="850" alt="image" src="https://github.com/user-attachments/assets/7c10c3e5-920b-4ad5-b165-c004f858ac14" />

Built the same multi-tool agent system using 4 different frameworks to understand their strengths and trade-offs.

---

##  What It Does

An AI agent that can:
-  Search the web (DuckDuckGo)
-  Look up Wikipedia articles
-  Find research papers (ArXiv)
-  Query employee database (SQLite)

Same tools, different approaches!

---

##  Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running

### Installation

```bash
# 1. Clone repository
git clone <your-repo>
cd multi-framework-agents

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install streamlit langchain langchain-community langgraph crewai pyautogen==0.2.18 ollama duckduckgo-search wikipedia arxiv sqlalchemy

# 4. Start Ollama
ollama serve  # In a new terminal

# 5. Download model
ollama pull llama2
```

### Run Any Framework

```bash
# Choose one:
streamlit run simple_multi_tool.py      # Fast & reliable
streamlit run langgraph_multi_tool.py   # Graph workflows
streamlit run crewai_multi_tool.py      # Team collaboration
streamlit run autogen_multi_tool.py     # Autonomous agents
```

---

##  Framework Comparison

| Framework | Speed | Complexity | Best For |
|-----------|-------|------------|----------|
| **Simple** | 6s | Low | Production apps |
| **LangGraph** | 12s | Medium | Complex workflows |
| **CrewAI** | 25s | High | Quality outputs |
| **AutoGen** | 30s | Very High | Research projects |

---

##  The 4 Frameworks

### 1. Simple Direct 
**File:** `simple_multi_tool.py`

**How it works:**
```
Query → Keyword Match → Execute Tool → Format → Done
```

**Pros:** Fastest, most reliable (98% success rate), production-ready  
**Cons:** Basic keyword matching, single-tool queries only

**Use when:** You need speed and reliability

---

### 2. LangGraph 
**File:** `langgraph_multi_tool.py`

**How it works:**
```
Entry → Router Node → Tool Node → Generator Node → End
```

**Pros:** Visual workflows, state management, easy debugging  
**Cons:** More complex setup

**Use when:** You have multi-step workflows with state

---

### 3. CrewAI 
**File:** `crewai_multi_tool.py`

**How it works:**
```
Researcher Agent → Analyst Agent → Writer Agent → Response
```

**The Crew:**
- **Researcher** - Finds information using all tools
- **Analyst** - Analyzes and organizes data
- **Writer** - Creates polished response

**Pros:** High quality outputs, role-based agents, collaborative  
**Cons:** Slower, more token usage

**Use when:** Quality matters more than speed

---

### 4. AutoGen 
**File:** `autogen_multi_tool.py`

**How it works:**
```
User → Coordinator → Specialists → Group Discussion → Response
```

**The Team:**
- **Coordinator** - Delegates to specialists
- **WebSearcher** - Web search specialist
- **WikipediaExpert** - Explanation specialist
- **ResearchSpecialist** - Paper specialist
- **DatabaseAnalyst** - SQL specialist

**Pros:** Autonomous, self-organizing, research-grade  
**Cons:** Slowest, most complex, unpredictable

**Use when:** Experimenting with multi-agent systems

---

## Try It Out

Test the same query in all frameworks:

```
"Find research papers on neural networks"
```

**What you'll see:**
- **Simple** (~8s): Direct ArXiv search
- **LangGraph** (~12s): Shows graph execution path
- **CrewAI** (~25s): 3 agents collaborate
- **AutoGen** (~30s): Agents discuss in group

---


## Example Queries

### Web Search
```
"What's the latest AI news?"
"Current Bitcoin price"
```

### Wikipedia
```
"Explain quantum computing"
"Tell me about Albert Einstein"
```

### ArXiv Papers
```
"Find papers on transformers"
"Research on computer vision"
```

### Database
```
"SELECT * FROM employees"
"SELECT AVG(salary) FROM employees GROUP BY department"
```

## Performance Tips

**Make it faster:**
1. Use smaller model: `ollama pull phi`
2. Reduce max iterations in agent configs
3. Use Simple framework for production

**Make it better:**
1. Use CrewAI for quality
2. Use larger model: `ollama pull llama2:13b`
3. Add more context in prompts

---

## Which Framework Should I Use?

**Choose based on your needs:**

```
Need it fast and reliable?
  └─ Use Simple 

Need complex workflows?
  └─ Use LangGraph 

Need high quality outputs?
  └─ Use CrewAI 

Doing research or experiments?
  └─ Use AutoGen 
```

---

## Requirements

```txt
streamlit==1.28.0
langchain==0.1.0
langchain-community==0.0.10
langchain-core==0.1.0
langgraph==0.0.20
crewai==0.1.25
pyautogen==0.2.18
ollama==0.1.6
duckduckgo-search==4.1.1
wikipedia==1.4.0
arxiv==2.0.0
sqlalchemy==2.0.23
```

---

## What I Learned

- Different approaches to same problem
- Trade-offs: speed vs quality vs complexity
- When to use each framework
- Multi-agent system design
- Tool integration patterns
- Production vs research patterns

---


*Comparing frameworks so you don't have to!*
