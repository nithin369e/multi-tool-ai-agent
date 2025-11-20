# Simplified Multi-Tool AI Agent - More Reliable
# pip install streamlit langchain langchain-community duckduckgo-search wikipedia arxiv pypdf2 python-docx sqlalchemy ollama

import streamlit as st
from datetime import datetime
import os
import sqlite3
import json

# LangChain imports
from langchain_community.llms import Ollama

# Tool imports
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

# Page configuration
st.set_page_config(
    page_title="Multi-Tool AI Agent",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #1e1e1e;
    }
    .tool-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .web-search { background-color: #4CAF50; color: white; }
    .wikipedia { background-color: #2196F3; color: white; }
    .arxiv { background-color: #FF9800; color: white; }
    .database { background-color: #9C27B0; color: white; }
    .decision-box {
        background-color: #2b2b2b;
        border-left: 3px solid #FFA500;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9em;
        color: #FFA500;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize SQLite Database
def init_database():
    """Initialize SQLite database with sample data"""
    conn = sqlite3.connect('agent_database.db')
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
            (4, 'Alice Williams', 'Sales', 85000, '2021-09-01'),
            (5, 'Charlie Brown', 'HR', 65000, '2022-02-14')
        ]
        cursor.executemany('INSERT INTO employees VALUES (?,?,?,?,?)', employees)
        conn.commit()
    
    conn.close()

# Tool functions
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results[:1000]  # Limit to 1000 chars
    except Exception as e:
        return f"Web search error: {str(e)}"

def wikipedia_search(query: str) -> str:
    """Search Wikipedia"""
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        results = wikipedia.run(query)
        return results[:1000]  # Limit to 1000 chars
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

def arxiv_search(query: str) -> str:
    """Search ArXiv for research papers"""
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        results = arxiv.run(query)
        return results[:1500]  # Limit to 1500 chars
    except Exception as e:
        return f"ArXiv error: {str(e)}"

def database_query(query: str) -> str:
    """Execute SQL queries"""
    try:
        if not query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries allowed"
        
        conn = sqlite3.connect('agent_database.db')
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        if not results:
            return "No results found"
        
        formatted = f"Columns: {', '.join(columns)}\n\n"
        for row in results:
            formatted += " | ".join(str(item) for item in row) + "\n"
        
        return formatted
    except Exception as e:
        return f"Database error: {str(e)}"

# Simple decision logic
def decide_tool(query: str):
    """Simple keyword-based tool selection"""
    query_lower = query.lower()
    
    # ArXiv keywords
    if any(word in query_lower for word in ['paper', 'research', 'arxiv', 'study', 'publication']):
        return 'arxiv', 'Query mentions research papers'
    
    # Database keywords
    if any(word in query_lower for word in ['select', 'employee', 'database', 'sql', 'show all', 'list all']):
        return 'database', 'Query is about database/employees'
    
    # Wikipedia keywords
    if any(word in query_lower for word in ['what is', 'who is', 'tell me about', 'explain', 'define']):
        return 'wikipedia', 'Query seeks explanation/definition'
    
    # Web search for everything else
    return 'web_search', 'Default to web search for current info'

# Initialize components
@st.cache_resource
def initialize_llm(model_name):
    """Initialize Ollama LLM"""
    try:
        llm = Ollama(model=model_name, temperature=0.7)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama2"

# Initialize database
init_database()

# Initialize LLM
llm = initialize_llm(st.session_state.model_name)

# Sidebar
with st.sidebar:
    st.header("🛠️ Multi-Tool AI Agent")
    
    # Model selection
    model_input = st.text_input(
        "Model Name",
        value=st.session_state.model_name,
        help="Ollama model name"
    )
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    st.divider()
    
    # Available Tools
    st.subheader("🔧 Available Tools")
    st.markdown('<span class="tool-badge web-search">🌐 Web Search</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge wikipedia">📚 Wikipedia</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge arxiv">📄 ArXiv Papers</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge database">🗄️ Database</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Database info
    st.subheader("🗄️ Database Tables")
    st.info("""
    **employees**
    - name, department, salary, hire_date
    
    **Example:**
    SELECT * FROM employees
    """)
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Try These")
    examples = [
        "What's the latest AI news?",
        "Tell me about Python",
        "Find papers on neural networks",
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

# Main interface
st.title("🛠️ Simplified Multi-Tool AI Agent")
st.caption("Fast and reliable - Direct tool execution")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            if "tool_used" in message:
                with st.expander("🔧 Tool Used"):
                    st.markdown(f'<div class="decision-box">{message["tool_used"]}</div>', 
                              unsafe_allow_html=True)
        
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)
    
    # Get response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        tool_placeholder = st.container()
        
        try:
            # Decide which tool to use
            tool, reason = decide_tool(prompt)
            
            with st.spinner(f"🔍 Using {tool}..."):
                # Execute tool
                if tool == 'web_search':
                    tool_result = web_search(prompt)
                    tool_name = "🌐 Web Search"
                elif tool == 'wikipedia':
                    tool_result = wikipedia_search(prompt)
                    tool_name = "📚 Wikipedia"
                elif tool == 'arxiv':
                    tool_result = arxiv_search(prompt)
                    tool_name = "📄 ArXiv"
                elif tool == 'database':
                    tool_result = database_query(prompt)
                    tool_name = "🗄️ Database"
                else:
                    tool_result = web_search(prompt)
                    tool_name = "🌐 Web Search"
                
                # Use LLM to format the response
                if llm:
                    llm_prompt = f"""Based on the following information, provide a clear and helpful answer to the user's question.

User Question: {prompt}

Information from {tool_name}:
{tool_result}

Provide a concise, well-formatted answer:"""
                    
                    answer = llm.invoke(llm_prompt)
                else:
                    answer = f"**Results from {tool_name}:**\n\n{tool_result}"
                
                response_placeholder.markdown(answer)
            
            # Show tool used
            with tool_placeholder:
                with st.expander("🔧 Tool Used", expanded=False):
                    st.markdown(f'<div class="decision-box">**Tool:** {tool_name}<br>**Reason:** {reason}</div>', 
                              unsafe_allow_html=True)
            
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": response_timestamp,
                "tool_used": f"**Tool:** {tool_name}<br>**Reason:** {reason}"
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
            response_placeholder.markdown(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Welcome message
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""👋 **Welcome to the Simplified Multi-Tool Agent!**

This version is faster and more reliable with **direct tool execution**:

🌐 **Web Search** - Current news and information
📚 **Wikipedia** - Encyclopedia knowledge  
📄 **ArXiv** - Research papers
🗄️ **Database** - Employee data queries

**Try:**
- "What's the latest AI news?"
- "Find papers on neural networks"
- "Tell me about quantum computing"
- "SELECT * FROM employees"

No complex agent reasoning - just fast, direct results! 🚀""")
        st.caption("Ready! 🛠️")