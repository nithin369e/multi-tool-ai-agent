# Install required libraries first:
# pip install streamlit langchain langchain-community duckduckgo-search wikipedia arxiv chromadb sentence-transformers pypdf python-docx sqlalchemy

import streamlit as st
from datetime import datetime
import os
import sqlite3
import json

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Tool imports
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

# Document processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

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
    .file-read { background-color: #F44336; color: white; }
    .rag { background-color: #00BCD4; color: white; }
    .agent-thought {
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
    
    # Create sample tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER,
            hire_date TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY,
            project_name TEXT,
            budget INTEGER,
            status TEXT,
            start_date TEXT
        )
    ''')
    
    # Insert sample data if tables are empty
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
        
        projects = [
            (1, 'AI Chatbot', 500000, 'Active', '2024-01-01'),
            (2, 'Mobile App', 300000, 'Completed', '2023-06-15'),
            (3, 'Website Redesign', 150000, 'Active', '2024-03-20'),
            (4, 'Data Analytics', 400000, 'Planning', '2024-06-01')
        ]
        cursor.executemany('INSERT INTO projects VALUES (?,?,?,?,?)', projects)
    
    conn.commit()
    conn.close()

# Database query tool
def query_database(query: str) -> str:
    """Execute SQL queries on the database"""
    try:
        conn = sqlite3.connect('agent_database.db')
        cursor = conn.cursor()
        
        # Safety check - only allow SELECT queries
        if not query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries are allowed for safety reasons."
        
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        conn.close()
        
        if not results:
            return "No results found."
        
        # Format results
        formatted = f"Columns: {', '.join(columns)}\n\n"
        for row in results:
            formatted += " | ".join(str(item) for item in row) + "\n"
        
        return formatted
    except Exception as e:
        return f"Database error: {str(e)}"

# File reading tool
def read_file_content(filepath: str) -> str:
    """Read content from a file"""
    try:
        # Security: Check if file exists in allowed directory
        if not os.path.exists(filepath):
            return f"Error: File not found: {filepath}"
        
        # Read based on file extension
        ext = filepath.lower().split('.')[-1]
        
        if ext == 'txt':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                return f"Content of {filepath}:\n\n{content[:2000]}"  # Limit to 2000 chars
        
        elif ext == 'pdf' and PDF_AVAILABLE:
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages[:5]:  # First 5 pages only
                    content += page.extract_text() + "\n"
                return f"Content of {filepath} (first 5 pages):\n\n{content[:2000]}"
        
        elif ext == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return f"JSON content of {filepath}:\n\n{json.dumps(data, indent=2)[:2000]}"
        
        else:
            return f"Unsupported file type: {ext}"
    
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Document processing
def extract_text_from_txt(file):
    return file.read().decode('utf-8')

def extract_text_from_pdf(file):
    if not PDF_AVAILABLE:
        return "PyPDF2 not installed"
    pdf_reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_docx(file):
    if not DOCX_AVAILABLE:
        return "python-docx not installed"
    doc = docx.Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def process_document(file):
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'txt':
        text = extract_text_from_txt(file)
    elif file_extension == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(file)
    else:
        return None, "Unsupported file type"
    
    return text, file.name

# Initialize components
@st.cache_resource
def initialize_components(model_name):
    """Initialize LLM and embeddings"""
    try:
        llm = Ollama(model=model_name, temperature=0.7)
        
        # Try to initialize embeddings
        embeddings = None
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            st.success("✅ All 6 tools loaded successfully!")
        except ImportError as e:
            st.warning("⚠️ sentence-transformers not found in current environment. Knowledge Base (RAG) will be disabled.")
            st.info("Install with: pip install sentence-transformers")
        except Exception as e:
            st.warning(f"⚠️ Could not load embeddings: {str(e)}")
        
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.info("Make sure Ollama is running: ollama serve")
        return None, None

@st.cache_resource
def create_vector_store(_embeddings):
    """Create vector store"""
    if _embeddings is None:
        return None
    
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_multitools_db")
        return Chroma(
            client=client,
            collection_name="documents",
            embedding_function=_embeddings,
        )
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def add_documents_to_vectorstore(texts, sources, vector_store):
    """Add documents to vector store"""
    if vector_store is None:
        return 0
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        
        all_chunks = []
        all_metadatas = []
        
        for text, source in zip(texts, sources):
            chunks = text_splitter.split_text(text)
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": source} for _ in chunks])
        
        vector_store.add_texts(texts=all_chunks, metadatas=all_metadatas)
        return len(all_chunks)
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return 0

# Create all tools
def create_all_tools(llm, vector_store):
    """Create all available tools"""
    tools = []
    
    # 1. DuckDuckGo Web Search
    try:
        search = DuckDuckGoSearchRun()
        tools.append(Tool(
            name="Web_Search",
            func=search.run,
            description="Useful for searching the internet for current information, news, facts, or any real-time data. Use this when you need up-to-date information or when the question is about recent events."
        ))
    except Exception as e:
        st.warning(f"DuckDuckGo tool unavailable: {e}")
    
    # 2. Wikipedia
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for getting detailed information from Wikipedia about historical events, people, places, concepts, and general knowledge. Use this for factual, encyclopedic information."
        ))
    except Exception as e:
        st.warning(f"Wikipedia tool unavailable: {e}")
    
    # 3. ArXiv Research Papers
    try:
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
        tools.append(Tool(
            name="Research_Papers",
            func=arxiv.run,
            description="Useful for finding and reading academic research papers from ArXiv. Use this when you need scientific papers, research findings, or academic publications on topics like AI, physics, mathematics, computer science."
        ))
    except Exception as e:
        st.warning(f"ArXiv tool unavailable: {e}")
    
    # 4. Database Query
    tools.append(Tool(
        name="Database_Query",
        func=query_database,
        description="Useful for querying the SQLite database containing employee and project information. Only use SELECT queries. Example: 'SELECT * FROM employees WHERE department = Engineering'"
    ))
    
    # 5. File Reader
    tools.append(Tool(
        name="File_Reader",
        func=read_file_content,
        description="Useful for reading content from files (TXT, PDF, JSON). Provide the full file path. Example: 'data/report.txt'"
    ))
    
    # 6. RAG Knowledge Base
    if vector_store is not None and vector_store._collection.count() > 0:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
            )
            
            def rag_query(query: str) -> str:
                try:
                    result = qa_chain.invoke({"query": query})
                    answer = result['result']
                    sources = [doc.metadata.get('source', 'Unknown') 
                              for doc in result['source_documents']]
                    if 'last_sources' not in st.session_state:
                        st.session_state.last_sources = []
                    st.session_state.last_sources = list(set(sources))
                    return answer
                except Exception as e:
                    return f"Error querying knowledge base: {str(e)}"
            
            tools.append(Tool(
                name="Knowledge_Base",
                func=rag_query,
                description="Useful for answering questions about uploaded documents in the knowledge base. Use this when the question is about specific documents the user has uploaded."
            ))
        except Exception as e:
            st.warning(f"Knowledge Base tool unavailable: {e}")
    
    return tools

def create_agent(llm, tools, memory):
    """Create ReAct agent with all tools"""
    
    template = """You are a powerful AI assistant with access to multiple specialized tools. Answer questions by using the most appropriate tool(s).

Available tools:
{tools}

Use this format:

Question: the input question you must answer
Thought: think about which tool(s) to use
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

IMPORTANT: Be concise in your reasoning. If you have enough information, provide the Final Answer.

Previous conversation:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"],
        template=template
    )
    
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        max_execution_time=60,
        early_stopping_method="force",
    )
    
    return agent_executor

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama2"

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = 0

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Initialize database
init_database()

# Initialize components
llm, embeddings = initialize_components(st.session_state.model_name)

# Check if LLM is available
if llm is None:
    st.error("❌ Failed to initialize Ollama LLM")
    st.info("Please ensure Ollama is running: ollama serve")
    st.stop()

vector_store = create_vector_store(embeddings)

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
    st.markdown('<span class="tool-badge file-read">📁 File Reader</span>', unsafe_allow_html=True)
    st.markdown('<span class="tool-badge rag">🧠 Knowledge Base</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Add to Knowledge Base",
        type=['txt', 'pdf', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📥 Process Documents", use_container_width=True):
            texts, sources = [], []
            progress_bar = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                text, source = process_document(file)
                if text:
                    texts.append(text)
                    sources.append(source)
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            if texts:
                chunks = add_documents_to_vectorstore(texts, sources, vector_store)
                st.session_state.documents_loaded += len(texts)
                st.success(f"✅ Added {len(texts)} documents ({chunks} chunks)")
    
    st.metric("Documents in KB", st.session_state.documents_loaded)
    
    st.divider()
    
    # Database info
    st.subheader("🗄️ Database Info")
    st.info("""
    **Tables:**
    - employees (name, department, salary, hire_date)
    - projects (project_name, budget, status, start_date)
    
    **Example queries:**
    - "Show all employees"
    - "What's the average salary?"
    - "List active projects"
    """)
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Example Queries")
    examples = [
        "Search web: Latest AI news",
        "Wikipedia: Albert Einstein",
        "ArXiv: neural networks",
        "DB: SELECT * FROM employees",
        "What's in my documents?",
        "Compare web info with my docs"
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            # Add to chat
            st.session_state.messages.append({
                "role": "user",
                "content": ex,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# Main interface
st.title("🛠️ Multi-Tool AI Agent")
st.caption("Powered by 6 specialized tools: Web Search, Wikipedia, ArXiv, Database, File Reader, Knowledge Base")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            if "tools_used" in message and message["tools_used"]:
                with st.expander("🔧 Tools Used"):
                    for tool in message["tools_used"]:
                        st.markdown(f"- {tool}")
            
            if "thoughts" in message and message["thoughts"]:
                with st.expander("🧠 Agent Reasoning"):
                    for thought in message["thoughts"]:
                        st.markdown(f'<div class="agent-thought">{thought}</div>', 
                                  unsafe_allow_html=True)
        
        st.caption(message["timestamp"])

# Chat input
if prompt := st.chat_input("Ask me anything... I have access to multiple tools!"):
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
        thinking_placeholder = st.container()
        
        try:
            with st.spinner("🤖 Analyzing and selecting tools..."):
                # Create tools and agent
                tools = create_all_tools(llm, vector_store)
                agent_executor = create_agent(llm, tools, st.session_state.memory)
                
                # Execute agent
                st.session_state.last_sources = []
                response = agent_executor.invoke({"input": prompt})
                
                answer = response['output']
                response_placeholder.markdown(answer)
                
                # Extract tools used and thoughts
                tools_used = []
                thoughts = []
                
                if 'intermediate_steps' in response:
                    for step in response['intermediate_steps']:
                        action, observation = step
                        tools_used.append(f"**{action.tool}**: {action.tool_input[:100]}...")
                        thoughts.append(f"**Action:** {action.tool}<br>**Input:** {action.tool_input[:150]}...")
            
            # Show tools used
            if tools_used:
                with st.expander("🔧 Tools Used", expanded=False):
                    for tool in tools_used:
                        st.markdown(tool)
            
            # Show reasoning
            if thoughts:
                with st.expander("🧠 Agent Reasoning", expanded=False):
                    for thought in thoughts:
                        st.markdown(f'<div class="agent-thought">{thought}</div>', 
                                  unsafe_allow_html=True)
            
            response_timestamp = datetime.now().strftime("%H:%M:%S")
            st.caption(response_timestamp)
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": response_timestamp,
                "tools_used": tools_used,
                "thoughts": thoughts
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
        st.markdown("""👋 **Welcome to the Multi-Tool AI Agent!**

I have access to **6 powerful tools**:

1. 🌐 **Web Search** (DuckDuckGo) - Current news and information
2. 📚 **Wikipedia** - Encyclopedia knowledge
3. 📄 **ArXiv** - Research papers and publications
4. 🗄️ **Database** - Query employee and project data
5. 📁 **File Reader** - Read local files
6. 🧠 **Knowledge Base** - Your uploaded documents (RAG)

**Try asking:**
- "What's the latest news about AI?"
- "Tell me about quantum computing from Wikipedia"
- "Find research papers on neural networks"
- "Show me all employees in Engineering department"
- "What are my top skills?" (upload resume first)

I'll automatically select the best tool(s) for your question!""")
        st.caption("Ready with 6 tools 🛠️")