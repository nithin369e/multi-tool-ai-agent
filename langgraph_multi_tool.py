# LangGraph Multi-Tool AI Agent
# pip install streamlit langgraph langchain langchain-community duckduckgo-search wikipedia arxiv ollama

import streamlit as st
from datetime import datetime
import sqlite3
from typing import TypedDict, Annotated, Sequence
import operator

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Page config
st.set_page_config(
    page_title="LangGraph Multi-Tool Agent",
    page_icon="🔷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; }
    .node-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
        background-color: #2196F3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
def init_database():
    conn = sqlite3.connect('langgraph_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            hire_date TEXT
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        employees = [
            (1, 'John Doe', 'Engineering', 95000, '2020-01-15'),
            (2, 'Jane Smith', 'Marketing', 75000, '2019-06-20'),
            (3, 'Bob Johnson', 'Engineering', 105000, '2018-03-10'),
        ]
        cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?)', employees)
        conn.commit()
    conn.close()

# Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_tool: str
    tool_result: str
    final_response: str
    query: str

# Tool functions
def web_search_tool(query: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

def wikipedia_tool(query: str) -> str:
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

def arxiv_tool(query: str) -> str:
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        return arxiv.run(query)[:1500]
    except Exception as e:
        return f"Error: {str(e)}"

def database_tool(query: str) -> str:
    try:
        if not query.strip().upper().startswith('SELECT'):
            return "Only SELECT queries allowed"
        conn = sqlite3.connect('langgraph_db.db')
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        if not results:
            return "No results"
        formatted = f"Columns: {', '.join(columns)}\n"
        for row in results:
            formatted += " | ".join(str(item) for item in row) + "\n"
        return formatted
    except Exception as e:
        return f"Error: {str(e)}"

# Graph nodes
def router_node(state: AgentState) -> AgentState:
    """Route to appropriate tool based on query"""
    query = state["query"].lower()
    
    if any(word in query for word in ['paper', 'research', 'arxiv']):
        state["current_tool"] = "arxiv"
    elif any(word in query for word in ['select', 'database', 'employee']):
        state["current_tool"] = "database"
    elif any(word in query for word in ['explain', 'what is', 'tell me']):
        state["current_tool"] = "wikipedia"
    else:
        state["current_tool"] = "web_search"
    
    state["messages"].append(AIMessage(content=f"Selected tool: {state['current_tool']}"))
    return state

def tool_executor_node(state: AgentState) -> AgentState:
    """Execute the selected tool"""
    query = state["query"]
    tool = state["current_tool"]
    
    if tool == "web_search":
        result = web_search_tool(query)
    elif tool == "wikipedia":
        result = wikipedia_tool(query)
    elif tool == "arxiv":
        result = arxiv_tool(query)
    elif tool == "database":
        result = database_tool(query)
    else:
        result = "Unknown tool"
    
    state["tool_result"] = result
    state["messages"].append(AIMessage(content=f"Tool result: {result[:200]}..."))
    return state

def response_generator_node(state: AgentState, llm) -> AgentState:
    """Generate final response using LLM"""
    query = state["query"]
    tool_result = state["tool_result"]
    
    prompt = f"""Based on the following information, provide a clear answer.

User Question: {query}
Tool Result: {tool_result}

Answer:"""
    
    try:
        response = llm.invoke(prompt)
        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))
    except Exception as e:
        state["final_response"] = f"Error: {str(e)}"
    
    return state

# Build the graph
def create_agent_graph(llm):
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("response_generator", lambda state: response_generator_node(state, llm))
    
    # Add edges
    workflow.set_entry_point("router")
    workflow.add_edge("router", "tool_executor")
    workflow.add_edge("tool_executor", "response_generator")
    workflow.add_edge("response_generator", END)
    
    return workflow.compile()

# Initialize
@st.cache_resource
def initialize_llm(model_name):
    return Ollama(model=model_name, temperature=0.7)

init_database()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama2"

llm = initialize_llm(st.session_state.model_name)
graph = create_agent_graph(llm)

# Sidebar
with st.sidebar:
    st.header("🔷 LangGraph Agent")
    
    model_input = st.text_input("Model", value=st.session_state.model_name)
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    st.divider()
    st.subheader("🔧 Graph Structure")
    st.code("""
    Start
      ↓
    Router (Tool Selection)
      ↓
    Tool Executor
      ↓
    Response Generator
      ↓
    End
    """)
    
    st.divider()
    st.subheader("💡 Examples")
    examples = [
        "What's the latest AI news?",
        "Find papers on transformers",
        "Explain quantum computing",
        "SELECT * FROM employees"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.messages.append({
                "role": "user",
                "content": ex,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()
    
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("🔷 LangGraph Multi-Tool Agent")
st.caption("State-based graph workflow with intelligent routing")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "graph_path" in message:
            with st.expander("🔷 Graph Execution Path"):
                for node in message["graph_path"]:
                    st.markdown(f'<span class="node-badge">{node}</span>', unsafe_allow_html=True)
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        try:
            with st.spinner("🔷 Executing graph..."):
                # Execute graph
                initial_state = {
                    "messages": [HumanMessage(content=prompt)],
                    "current_tool": "",
                    "tool_result": "",
                    "final_response": "",
                    "query": prompt
                }
                
                result = graph.invoke(initial_state)
                answer = result["final_response"]
                tool_used = result["current_tool"]
                
                response_placeholder.markdown(answer)
                
                # Show graph execution
                with st.expander("🔷 Graph Execution", expanded=False):
                    st.markdown(f"**Path:** Router → Tool Executor ({tool_used}) → Response Generator")
                    st.markdown(f'<span class="node-badge">Router</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="node-badge">Tool: {tool_used}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="node-badge">Generator</span>', unsafe_allow_html=True)
            
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": response_timestamp,
                "graph_path": ["Router", f"Tool: {tool_used}", "Generator"]
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Welcome
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""👋 **Welcome to LangGraph Multi-Tool Agent!**

**What's Different:**
- 🔷 **State-based graph workflow**
- 📊 **Visual execution paths**
- 🔄 **Node-by-node processing**
- 🎯 **Deterministic routing**

**Try:**
- "What's the latest AI news?"
- "Find papers on neural networks"
- "Explain machine learning"

LangGraph enables complex workflows with clear state management! 🚀""")
        st.caption("Graph ready 🔷")