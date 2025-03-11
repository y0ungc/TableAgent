"""
Configuration settings for the table processing agent.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-3.5-turbo"  # Default model

# Agent configuration
MAX_TOKENS = 1000
TEMPERATURE = 0.3

# Pandas display options
MAX_ROWS_DISPLAY = 20
MAX_COLS_DISPLAY = 10

# Logging configuration
LOG_LEVEL = "INFO"

# Default prompts
SYSTEM_PROMPT = """
You are a table processing assistant that helps users analyze and manipulate tabular data.
Your task is to understand the user's request and execute the appropriate pandas operations.
"""

REWRITE_PROMPT = """
Based on the conversation history and the current dataframe, rewrite the following user query 
to be more specific and executable using pandas operations:

User query: {query}

Current dataframe information:
{dataframe_info}

Conversation history:
{conversation_history}

Rewritten query:
"""

TOOL_SELECTION_PROMPT = """
Given the following user request and available tools, determine which tool or sequence of tools 
should be used to fulfill the request.

User request: {query}

Available tools:
{tools}

Dataframe information:
{dataframe_info}

Provide your reasoning and the selected tool(s):
"""