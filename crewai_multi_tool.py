# CrewAI Multi-Tool Agent System
# pip install streamlit crewai crewai-tools langchain-community duckduckgo-search wikipedia arxiv ollama

import streamlit as st
from datetime import datetime
import sqlite3

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.tools import tool

# Page config
st.set_page_config(
    page_title="CrewAI Multi-Tool System",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; }
    .agent-badge {
        display: inline-block;
        padding: 5px 10px;
        margin: 5px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .researcher { background-color: #4CAF50; color: white; }
    .analyst { background-color: #2196F3; color: white; }
    .writer { background-color: #FF9800; color: white; }
    </style>
""", unsafe_allow_html=True)

# Initialize database
def init_database():
    conn = sqlite3.connect('crewai_db.db')
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

# Define tools using @tool decorator
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Search the web for current information using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

@tool("Wikipedia Tool")
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for encyclopedic knowledge"""
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

@tool("ArXiv Tool")
def arxiv_tool(query: str) -> str:
    """Search ArXiv for research papers"""
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        return arxiv.run(query)[:1500]
    except Exception as e:
        return f"Error: {str(e)}"

@tool("Database Tool")
def database_tool(query: str) -> str:
    """Execute SQL queries on the employee database"""
    try:
        if not query.strip().upper().startswith('SELECT'):
            return "Only SELECT queries allowed"
        conn = sqlite3.connect('crewai_db.db')
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

# Create agents
def create_agents(llm):
    """Create specialized agents for the crew"""
    
    # Research Agent
    researcher = Agent(
        role='Research Specialist',
        goal='Find accurate and relevant information from various sources',
        backstory="""You are an expert researcher with access to web search, 
        Wikipedia, ArXiv papers, and databases. You excel at finding the most 
        relevant information for any query.""",
        tools=[web_search_tool, wikipedia_tool, arxiv_tool, database_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Analysis Agent
    analyst = Agent(
        role='Information Analyst',
        goal='Analyze and synthesize information from multiple sources',
        backstory="""You are a skilled analyst who can take raw information 
        and extract key insights, patterns, and meaningful conclusions.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    # Writer Agent
    writer = Agent(
        role='Content Writer',
        goal='Create clear, concise, and well-formatted responses',
        backstory="""You are an expert writer who transforms complex 
        information into easy-to-understand, well-structured responses.""",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )
    
    return researcher, analyst, writer

# Create tasks
def create_tasks(query, researcher, analyst, writer):
    """Create tasks for the crew"""
    
    research_task = Task(
        description=f"""Research the following query using appropriate tools:
        Query: {query}
        
        Use web search for current information, Wikipedia for explanations,
        ArXiv for research papers, or database for SQL queries.
        Provide comprehensive information.""",
        agent=researcher,
        expected_output="Detailed information from relevant sources"
    )
    
    analysis_task = Task(
        description="""Analyze the research findings and extract key insights.
        Identify the most important information and organize it logically.""",
        agent=analyst,
        expected_output="Organized and analyzed information",
        context=[research_task]
    )
    
    writing_task = Task(
        description="""Write a clear, concise response based on the analysis.
        Format it well and make it easy to understand.""",
        agent=writer,
        expected_output="Well-formatted final response",
        context=[analysis_task]
    )
    
    return [research_task, analysis_task, writing_task]

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

# Sidebar
with st.sidebar:
    st.header("👥 CrewAI System")
    
    model_input = st.text_input("Model", value=st.session_state.model_name)
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    st.divider()
    st.subheader("👥 The Crew")
    st.markdown('<span class="agent-badge researcher">🔍 Researcher</span>', unsafe_allow_html=True)
    st.caption("Finds information using tools")
    
    st.markdown('<span class="agent-badge analyst">📊 Analyst</span>', unsafe_allow_html=True)
    st.caption("Analyzes and synthesizes data")
    
    st.markdown('<span class="agent-badge writer">✍️ Writer</span>', unsafe_allow_html=True)
    st.caption("Creates final response")
    
    st.divider()
    st.subheader("🔄 Workflow")
    st.code("""
    Researcher
        ↓
    Analyst  
        ↓
    Writer
        ↓
    Final Response
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
st.title("👥 CrewAI Multi-Tool System")
st.caption("Collaborative agents working together for better answers")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "crew_process" in message:
            with st.expander("👥 Crew Process"):
                for step in message["crew_process"]:
                    st.markdown(step)
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask the crew..."):
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
            with st.spinner("👥 Crew is working..."):
                # Create crew
                researcher, analyst, writer = create_agents(llm)
                tasks = create_tasks(prompt, researcher, analyst, writer)
                
                crew = Crew(
                    agents=[researcher, analyst, writer],
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True
                )
                
                # Execute crew
                result = crew.kickoff()
                answer = str(result)
                
                response_placeholder.markdown(answer)
                
                # Show crew process
                with st.expander("👥 Crew Collaboration", expanded=False):
                    st.markdown("**Process:**")
                    st.markdown("1. 🔍 **Researcher** gathered information")
                    st.markdown("2. 📊 **Analyst** synthesized data")
                    st.markdown("3. ✍️ **Writer** formatted response")
            
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": response_timestamp,
                "crew_process": [
                    "🔍 Researcher: Gathered information",
                    "📊 Analyst: Synthesized data",
                    "✍️ Writer: Formatted response"
                ]
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
        st.markdown("""👋 **Welcome to CrewAI Multi-Tool System!**

**Meet Your Crew:**
- 🔍 **Researcher** - Finds information using 4 tools
- 📊 **Analyst** - Analyzes and synthesizes data
- ✍️ **Writer** - Creates polished responses

**Collaborative Approach:**
Each agent specializes in their role, working together 
sequentially to provide comprehensive answers!

**Try:**
- "What's the latest AI news?"
- "Find papers on neural networks"
- "Explain machine learning"

Experience the power of collaborative AI! 🚀""")
        st.caption("Crew ready 👥")