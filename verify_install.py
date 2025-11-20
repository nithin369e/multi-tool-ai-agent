#!/usr/bin/env python3
"""Verify all frameworks are installed correctly"""

def check_simple():
    try:
        import streamlit
        from langchain_community.llms import Ollama
        print("✅ Simple framework ready")
        return True
    except ImportError as e:
        print(f"❌ Simple framework error: {e}")
        return False

def check_langgraph():
    try:
        from langgraph.graph import StateGraph
        print("✅ LangGraph framework ready")
        return True
    except ImportError as e:
        print(f"❌ LangGraph framework error: {e}")
        return False

def check_crewai():
    try:
        from crewai import Agent, Task, Crew
        print("✅ CrewAI framework ready")
        return True
    except ImportError as e:
        print(f"❌ CrewAI framework error: {e}")
        return False

def check_autogen():
    try:
        import autogen
        print("✅ AutoGen framework ready")
        return True
    except ImportError as e:
        print(f"❌ AutoGen framework error: {e}")
        return False

if __name__ == "__main__":
    print("Checking framework installations...\n")
    
    simple = check_simple()
    langgraph = check_langgraph()
    crewai = check_crewai()
    autogen = check_autogen()
    
    print(f"\nSummary: {sum([simple, langgraph, crewai, autogen])}/4 frameworks ready")