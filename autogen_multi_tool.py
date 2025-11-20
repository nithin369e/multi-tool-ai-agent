# AutoGen Multi-Tool Agent System
# pip install streamlit pyautogen langchain-community duckduckgo-search wikipedia arxiv ollama

import streamlit as st
from datetime import datetime
import sqlite3
from typing import Dict, List

# AutoGen imports
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    AUTOGEN_AVAILABLE = True
    # Note: AutoGen 0.10+ has a completely different API
    # This version may not work as expected with the new API
    st.warning("⚠️ AutoGen 0.10+ detected. This app was designed for AutoGen 0.2.x. Some features may not work.")
except ImportError:
    try:
        import autogen
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False
        st.error("AutoGen not installed or incompatible version.")
        st.info("For this app, install: pip install 'pyautogen<0.3'")
        st.stop()

# Tool imports
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# Page config
st.set_page_config(
    page_title="AutoGen Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #1e1e1e; }
    .agent-card {
        background-color: #2b2b2b;
        border-left: 3px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .conversation-flow {
        background-color: #2b2b2b;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
def init_database():
    conn = sqlite3.connect('autogen_db.db')
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

# Tool functions
def web_search_function(query: str) -> str:
    """Search the web using DuckDuckGo"""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

def wikipedia_function(query: str) -> str:
    """Search Wikipedia"""
    try:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

def arxiv_function(query: str) -> str:
    """Search ArXiv for papers"""
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        return arxiv.run(query)[:1500]
    except Exception as e:
        return f"Error: {str(e)}"

def database_function(query: str) -> str:
    """Execute SQL queries"""
    try:
        if not query.strip().upper().startswith('SELECT'):
            return "Only SELECT queries allowed"
        conn = sqlite3.connect('autogen_db.db')
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

# Configure LLM
def get_llm_config(model_name: str):
    """Get LLM configuration for AutoGen"""
    return {
        "config_list": [{
            "model": model_name,
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",  # Ollama doesn't need real API key
        }],
        "temperature": 0.7,
    }

# Create AutoGen agents
def create_autogen_agents(model_name: str):
    """Create specialized AutoGen agents"""
    
    llm_config = get_llm_config(model_name)
    
    # User proxy (executes functions)
    user_proxy = UserProxyAgent(
        name="UserProxy",
        system_message="A proxy for the user to execute functions.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        function_map={
            "web_search": web_search_function,
            "wikipedia_search": wikipedia_function,
            "arxiv_search": arxiv_function,
            "database_query": database_function,
        }
    )
    
    # Coordinator agent
    coordinator = AssistantAgent(
        name="Coordinator",
        system_message="""You are a coordinator who analyzes user queries and 
        delegates to appropriate specialists. You DON'T execute tools yourself.
        
        For web/news queries, ask WebSearcher.
        For explanations, ask WikipediaExpert.
        For research papers, ask ResearchSpecialist.
        For SQL queries, ask DatabaseAnalyst.""",
        llm_config=llm_config,
    )
    
    # Web search specialist
    web_searcher = AssistantAgent(
        name="WebSearcher",
        system_message="""You are a web search specialist. When asked to search,
        use the web_search function and provide results.""",
        llm_config=llm_config,
        function_map={"web_search": web_search_function}
    )
    
    # Wikipedia expert
    wiki_expert = AssistantAgent(
        name="WikipediaExpert",
        system_message="""You are a Wikipedia expert. When asked for explanations,
        use the wikipedia_search function and provide educational content.""",
        llm_config=llm_config,
        function_map={"wikipedia_search": wikipedia_function}
    )
    
    # Research specialist
    research_specialist = AssistantAgent(
        name="ResearchSpecialist",
        system_message="""You are a research paper specialist. When asked about
        papers, use the arxiv_search function and provide academic resources.""",
        llm_config=llm_config,
        function_map={"arxiv_search": arxiv_function}
    )
    
    # Database analyst
    db_analyst = AssistantAgent(
        name="DatabaseAnalyst",
        system_message="""You are a database analyst. When asked about data,
        use the database_query function with SQL and provide results.""",
        llm_config=llm_config,
        function_map={"database_query": database_function}
    )
    
    return user_proxy, coordinator, web_searcher, wiki_expert, research_specialist, db_analyst

# Initialize
init_database()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama2"

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar
with st.sidebar:
    st.header("🤖 AutoGen System")
    
    model_input = st.text_input("Model", value=st.session_state.model_name)
    if model_input != st.session_state.model_name:
        st.session_state.model_name = model_input
        st.rerun()
    
    st.divider()
    st.subheader("🤖 Agent Team")
    
    agents_info = [
        ("👔 Coordinator", "Delegates to specialists"),
        ("🌐 WebSearcher", "Handles web searches"),
        ("📚 WikipediaExpert", "Provides explanations"),
        ("📄 ResearchSpecialist", "Finds papers"),
        ("🗄️ DatabaseAnalyst", "Executes SQL")
    ]
    
    for name, desc in agents_info:
        st.markdown(f'<div class="agent-card"><b>{name}</b><br>{desc}</div>', 
                   unsafe_allow_html=True)
    
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
        st.session_state.conversation_history = []
        st.rerun()

# Main UI
st.title("🤖 AutoGen Multi-Agent System")
st.caption("Autonomous agents collaborating to solve tasks")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "agent_conversation" in message:
            with st.expander("🤖 Agent Conversation"):
                for conv in message["agent_conversation"]:
                    st.markdown(f'<div class="conversation-flow">{conv}</div>', 
                              unsafe_allow_html=True)
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask the team..."):
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
            with st.spinner("🤖 Agents are collaborating..."):
                # Create agents
                user_proxy, coordinator, web_searcher, wiki_expert, research_specialist, db_analyst = create_autogen_agents(st.session_state.model_name)
                
                # Create group chat
                groupchat = GroupChat(
                    agents=[user_proxy, coordinator, web_searcher, wiki_expert, research_specialist, db_analyst],
                    messages=[],
                    max_round=10
                )
                
                manager = GroupChatManager(groupchat=groupchat, llm_config=get_llm_config(st.session_state.model_name))
                
                # Initiate conversation
                user_proxy.initiate_chat(
                    manager,
                    message=prompt
                )
                
                # Get the final response
                chat_history = groupchat.messages
                final_response = chat_history[-1]["content"] if chat_history else "No response generated"
                
                response_placeholder.markdown(final_response)
                
                # Show agent conversation
                with st.expander("🤖 Agent Collaboration", expanded=False):
                    for msg in chat_history:
                        agent_name = msg.get("name", "Unknown")
                        content = msg.get("content", "")[:200]
                        st.markdown(f"**{agent_name}:** {content}...")
            
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            # Save conversation
            agent_conv = [f"**{msg.get('name', 'Unknown')}**: {msg.get('content', '')[:100]}..." 
                         for msg in chat_history]
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "timestamp": response_timestamp,
                "agent_conversation": agent_conv
            })
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}\n\nMake sure Ollama is running with API enabled."
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

# Welcome
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""👋 **Welcome to AutoGen Multi-Agent System!**

**Meet Your Agent Team:**
- 👔 **Coordinator** - Delegates tasks to specialists
- 🌐 **WebSearcher** - Handles web searches
- 📚 **WikipediaExpert** - Provides explanations  
- 📄 **ResearchSpecialist** - Finds papers
- 🗄️ **DatabaseAnalyst** - Executes SQL

**How It Works:**
Agents autonomously communicate and collaborate in a 
group chat to solve your query!

**Try:**
- "What's the latest AI news?"
- "Find papers on neural networks"
- "Explain machine learning"

Experience autonomous agent collaboration! 🚀""")
        st.caption("Team ready 🤖")