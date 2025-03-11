"""
LLM integration for the table processing agent.
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import openai
from datetime import datetime
from .config import OPENAI_API_KEY, LLM_MODEL, SYSTEM_PROMPT, REWRITE_PROMPT, TOOL_SELECTION_PROMPT

# Set up OpenAI API key
openai.api_key = OPENAI_API_KEY

class LLMIntegration:
    """
    LLM integration for the table processing agent.
    """
    def __init__(self, model: str = LLM_MODEL):
        """
        Initialize the LLM integration.
        
        Args:
            model (str, optional): LLM model to use. Defaults to LLM_MODEL from config.
        """
        self.model = model
        self.conversation_history = []
        
    def _add_to_history(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role (str): Role of the message sender ('user', 'assistant', 'system').
            content (str): Content of the message.
        """
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now()
        })
    
    def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Get a response from the LLM.
        
        Args:
            messages (List[Dict[str, str]]): List of messages to send to the LLM.
            
        Returns:
            str: LLM response.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message['content']
        except Exception as e:
            raise Exception(f"Error getting LLM response: {str(e)}")
    
    def rewrite_user_query(self, query: str, dataframe_info: str) -> str:
        """
        Rewrite a user query to be more specific and executable.
        
        Args:
            query (str): User query.
            dataframe_info (str): Information about the dataframe.
            
        Returns:
            str: Rewritten query.
        """
        # Convert conversation history to a string
        history_str = ""
        for msg in self.conversation_history[-5:]:  # Only use the last 5 messages
            if msg['role'] != 'system':  # Skip system messages
                history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
        
        # Format the prompt
        prompt = REWRITE_PROMPT.format(
            query=query,
            dataframe_info=dataframe_info,
            conversation_history=history_str
        )
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites user queries to be more specific and executable using pandas operations."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_llm_response(messages)
        
        # Add to conversation history
        self._add_to_history('user', query)
        self._add_to_history('assistant', f"I'll interpret that as: {response}")
        
        return response
    
    def select_tools(self, query: str, available_tools: List[str], dataframe_info: str) -> Dict[str, Any]:
        """
        Select the appropriate tools to fulfill a user request.
        
        Args:
            query (str): User query.
            available_tools (List[str]): List of available tools.
            dataframe_info (str): Information about the dataframe.
            
        Returns:
            Dict[str, Any]: Selected tools and parameters.
        """
        # Format the prompt
        tools_str = "\n".join([f"- {tool}" for tool in available_tools])
        prompt = TOOL_SELECTION_PROMPT.format(
            query=query,
            tools=tools_str,
            dataframe_info=dataframe_info
        )
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that selects the appropriate tools to fulfill user requests."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_llm_response(messages)
        
        # Parse the response to extract the selected tools and parameters
        # This is a simplified implementation; in a real-world scenario, you would want to use a more robust parsing method
        try:
            # Try to find a JSON object in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON object is found, try to parse the response manually
                lines = response.strip().split('\n')
                result = {}
                
                # Look for lines that contain tool names and parameters
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result[key.strip()] = value.strip()
                
                return result
        except Exception as e:
            # If parsing fails, return the raw response
            return {"raw_response": response, "error": str(e)}
    
    def generate_dynamic_plan(self, query: str, dataframe_info: str) -> List[Dict[str, Any]]:
        """
        Generate a dynamic plan to fulfill a complex user request.
        
        Args:
            query (str): User query.
            dataframe_info (str): Information about the dataframe.
            
        Returns:
            List[Dict[str, Any]]: List of steps to execute.
        """
        # Format the prompt
        prompt = f"""
        Given the following user request and dataframe information, generate a step-by-step plan to fulfill the request using pandas operations.
        
        User request: {query}
        
        Dataframe information:
        {dataframe_info}
        
        Please provide a detailed plan with the following structure:
        1. Step description
        2. Pandas operation to use
        3. Parameters for the operation
        
        Format your response as a JSON array of steps.
        """
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates step-by-step plans to fulfill user requests using pandas operations."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_llm_response(messages)
        
        # Try to parse the response as JSON
        try:
            # Find JSON array in the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON array is found, return a simple plan
                return [{"description": "Execute the request", "operation": "custom", "parameters": {"query": query}}]
        except Exception as e:
            # If parsing fails, return a simple plan
            return [{"description": "Execute the request", "operation": "custom", "parameters": {"query": query, "error": str(e)}}]
    
    def explain_result(self, result: Any, query: str, operations_performed: List[str]) -> str:
        """
        Generate an explanation of the result.
        
        Args:
            result (Any): Result of the operations.
            query (str): Original user query.
            operations_performed (List[str]): List of operations performed.
            
        Returns:
            str: Explanation of the result.
        """
        # Format the prompt
        operations_str = "\n".join([f"- {op}" for op in operations_performed])
        
        result_str = str(result)
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "... (truncated)"
        
        prompt = f"""
        Given the following user request, operations performed, and result, generate a clear and concise explanation of the result.
        
        User request: {query}
        
        Operations performed:
        {operations_str}
        
        Result:
        {result_str}
        
        Please provide a clear explanation that helps the user understand the result.
        """
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains the results of data operations to users."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_llm_response(messages)
        
        # Add to conversation history
        self._add_to_history('assistant', response)
        
        return response
    
    def handle_error(self, error: Exception, query: str) -> str:
        """
        Generate a helpful error message.
        
        Args:
            error (Exception): The error that occurred.
            query (str): Original user query.
            
        Returns:
            str: Helpful error message.
        """
        # Format the prompt
        prompt = f"""
        An error occurred while trying to fulfill the following user request:
        
        User request: {query}
        
        Error: {str(error)}
        
        Please provide a helpful error message that explains what went wrong and suggests possible solutions.
        """
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains errors to users and suggests solutions."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_llm_response(messages)
        
        # Add to conversation history
        self._add_to_history('assistant', response)
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the conversation history.
        
        Returns:
            List[Dict[str, Any]]: Conversation history.
        """
        return self.conversation_history 