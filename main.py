"""
Main script for the table processing agent.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
import json
import re
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utility modules
from utils.pandas_tools import PandasTools
from utils.llm_integration import LLMIntegration
from utils.config import OPENAI_API_KEY, MAX_ROWS_DISPLAY, MAX_COLS_DISPLAY

class TableAgent:
    """
    Agent for processing tables using LLM and pandas.
    """
    def __init__(self):
        """
        Initialize the table agent.
        """
        # Check if OpenAI API key is set
        if not OPENAI_API_KEY:
            print("Warning: OpenAI API key is not set. Please set it in the .env file.")
        
        # Initialize the pandas tools
        self.pandas_tools = PandasTools()
        
        # Initialize the LLM integration
        self.llm = LLMIntegration()
        
        # Set pandas display options
        pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
        pd.set_option('display.max_columns', MAX_COLS_DISPLAY)
        pd.set_option('display.width', 1000)
        
        # Available tools
        self.available_tools = self._get_available_tools()
        
        # Current dataframe info
        self.dataframe_info = "No dataframe loaded"
    
    def _get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the available tools.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of available tools.
        """
        # Get all methods from PandasTools class
        tools = {}
        for method_name in dir(self.pandas_tools):
            # Skip private methods and attributes
            if method_name.startswith('_') or not callable(getattr(self.pandas_tools, method_name)):
                continue
            
            # Get the method
            method = getattr(self.pandas_tools, method_name)
            
            # Get the docstring
            docstring = method.__doc__ or ""
            
            # Extract the description from the docstring
            description = docstring.strip().split('\n')[0] if docstring else ""
            
            # Add to tools dictionary
            tools[method_name] = {
                "description": description,
                "method": method
            }
        
        return tools
    
    def _update_dataframe_info(self):
        """
        Update the dataframe info.
        """
        if self.pandas_tools.df is not None:
            # Get basic dataframe info
            info = f"Dataframe shape: {self.pandas_tools.df.shape}\n"
            info += f"Columns: {list(self.pandas_tools.df.columns)}\n"
            
            # Get data types
            info += "Data types:\n"
            for col, dtype in self.pandas_tools.df.dtypes.items():
                info += f"  {col}: {dtype}\n"
            
            # Get sample data
            info += "Sample data (first 3 rows):\n"
            info += str(self.pandas_tools.df.head(3))
            
            self.dataframe_info = info
        else:
            self.dataframe_info = "No dataframe loaded"
    
    def load_data(self, file_path: str) -> str:
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            str: Success message.
        """
        try:
            self.pandas_tools.load_data(file_path)
            self._update_dataframe_info()
            return f"Successfully loaded data from {file_path}"
        except Exception as e:
            return f"Error loading data: {str(e)}"
    
    def process_request(self, request: str) -> str:
        """
        Process a user request.
        
        Args:
            request (str): User request.
            
        Returns:
            str: Response to the user.
        """
        try:
            # Check if a dataframe is loaded
            if self.pandas_tools.df is None and not request.lower().startswith("load"):
                return "No dataframe loaded. Please load a dataframe first."
            
            # Handle load requests directly
            if request.lower().startswith("load"):
                # Extract the file path
                match = re.search(r'load\s+(?:the\s+)?(?:file\s+)?(?:from\s+)?[\'"]?([^\'"]+)[\'"]?', request.lower())
                if match:
                    file_path = match.group(1)
                    return self.load_data(file_path)
                else:
                    return "Please specify a file path to load."
            
            # Rewrite the user query to be more specific
            rewritten_query = self.llm.rewrite_user_query(request, self.dataframe_info)
            
            # Determine the complexity of the request
            complexity = self._determine_complexity(rewritten_query)
            
            # Process the request based on its complexity
            if complexity == "simple":
                # Simple request (can be handled by a single tool)
                return self._process_simple_request(rewritten_query)
            elif complexity == "predefined_chain":
                # Predefined chain of tools
                return self._process_predefined_chain(rewritten_query)
            elif complexity == "dynamic_plan":
                # Dynamic plan
                return self._process_dynamic_plan(rewritten_query)
            else:
                # Fallback
                return self._process_fallback(rewritten_query)
        except Exception as e:
            # Handle errors
            traceback.print_exc()
            return self.llm.handle_error(e, request)
    
    def _determine_complexity(self, query: str) -> str:
        """
        Determine the complexity of a query.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Complexity level ('simple', 'predefined_chain', 'dynamic_plan', 'fallback').
        """
        # Get available tool names
        tool_names = list(self.available_tools.keys())
        
        # Format the prompt
        prompt = f"""
        Given the following user query and available tools, determine the complexity of the query.
        
        User query: {query}
        
        Available tools:
        {', '.join(tool_names)}
        
        Complexity levels:
        - simple: Can be handled by a single tool
        - predefined_chain: Can be handled by a predefined chain of tools
        - dynamic_plan: Requires a dynamic plan with multiple tools
        - fallback: Cannot be handled by the available tools
        
        Please respond with only one of the complexity levels.
        """
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that determines the complexity of user queries."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.get_llm_response(messages).strip().lower()
        
        # Extract the complexity level
        if "simple" in response:
            return "simple"
        elif "predefined_chain" in response:
            return "predefined_chain"
        elif "dynamic_plan" in response:
            return "dynamic_plan"
        else:
            return "fallback"
    
    def _process_simple_request(self, query: str) -> str:
        """
        Process a simple request that can be handled by a single tool.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Response to the user.
        """
        # Select the appropriate tool
        tool_selection = self.llm.select_tools(query, list(self.available_tools.keys()), self.dataframe_info)
        
        # Extract the tool name and parameters
        if "tool" in tool_selection:
            tool_name = tool_selection["tool"]
        elif "operation" in tool_selection:
            tool_name = tool_selection["operation"]
        else:
            # Try to find a tool name in the raw response
            for tool_name in self.available_tools.keys():
                if tool_name.lower() in str(tool_selection).lower():
                    break
            else:
                return "Could not determine the appropriate tool to use."
        
        # Check if the tool exists
        if tool_name not in self.available_tools:
            return f"Tool '{tool_name}' not found."
        
        # Extract parameters
        params = {}
        if "parameters" in tool_selection:
            params = tool_selection["parameters"]
        
        # Execute the tool
        try:
            tool_method = self.available_tools[tool_name]["method"]
            result = tool_method(**params)
            
            # Format the result
            if isinstance(result, (pd.DataFrame, pd.Series)):
                formatted_result = self.pandas_tools.format_output(result)
            elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], plt.Figure):
                # Handle the case where the result is a tuple of (dataframe, figure)
                df, fig = result
                # Save the figure to a temporary file
                fig_path = f"temp_figure_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                fig.savefig(fig_path)
                plt.close(fig)
                formatted_result = f"Figure saved to {fig_path}\n\n{self.pandas_tools.format_output(df)}"
            else:
                formatted_result = str(result)
            
            # Generate an explanation
            operations = [f"Used {tool_name} with parameters: {params}"]
            explanation = self.llm.explain_result(formatted_result, query, operations)
            
            return explanation
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def _process_predefined_chain(self, query: str) -> str:
        """
        Process a request that can be handled by a predefined chain of tools.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Response to the user.
        """
        # Define predefined chains
        predefined_chains = {
            "annual_report": {
                "description": "Generate an annual report",
                "steps": [
                    {"tool": "generate_annual_report", "params": {"date_column": None, "value_column": None}}
                ]
            },
            "monthly_report": {
                "description": "Generate a monthly report",
                "steps": [
                    {"tool": "generate_monthly_report", "params": {"date_column": None, "value_column": None, "year": None}}
                ]
            },
            "trend_analysis": {
                "description": "Analyze trends",
                "steps": [
                    {"tool": "analyze_trend", "params": {"date_column": None, "value_column": None, "freq": "M"}}
                ]
            },
            "correlation_analysis": {
                "description": "Analyze correlations",
                "steps": [
                    {"tool": "correlation_analysis", "params": {"columns": None}}
                ]
            }
        }
        
        # Select the appropriate chain
        prompt = f"""
        Given the following user query and predefined chains, select the most appropriate chain.
        
        User query: {query}
        
        Predefined chains:
        {json.dumps(predefined_chains, indent=2)}
        
        Please respond with the name of the most appropriate chain.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that selects the appropriate predefined chain for user queries."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.get_llm_response(messages).strip().lower()
        
        # Extract the chain name
        chain_name = None
        for name in predefined_chains.keys():
            if name.lower() in response:
                chain_name = name
                break
        
        if chain_name is None:
            return "Could not determine the appropriate predefined chain to use."
        
        # Get the chain
        chain = predefined_chains[chain_name]
        
        # Fill in the parameters
        prompt = f"""
        Given the following user query and predefined chain, fill in the parameters for the chain.
        
        User query: {query}
        
        Chain: {json.dumps(chain, indent=2)}
        
        Dataframe information:
        {self.dataframe_info}
        
        Please respond with a JSON object containing the filled parameters.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that fills in parameters for predefined chains."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.get_llm_response(messages)
        
        # Extract the parameters
        try:
            # Try to find a JSON object in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                params = json.loads(json_str)
            else:
                # If no JSON object is found, try to parse the response manually
                params = {}
                for step in chain["steps"]:
                    for param_name in step["params"].keys():
                        # Look for the parameter in the response
                        pattern = rf"{param_name}[:\s]+([^\n,]+)"
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            params[param_name] = match.group(1).strip()
        except Exception as e:
            return f"Error parsing parameters: {str(e)}"
        
        # Execute the chain
        operations = []
        results = []
        
        for step in chain["steps"]:
            tool_name = step["tool"]
            
            # Check if the tool exists
            if tool_name not in self.available_tools:
                return f"Tool '{tool_name}' not found."
            
            # Extract parameters for this step
            step_params = {}
            for param_name, param_value in step["params"].items():
                if param_name in params:
                    step_params[param_name] = params[param_name]
                else:
                    step_params[param_name] = param_value
            
            # Execute the tool
            try:
                tool_method = self.available_tools[tool_name]["method"]
                result = tool_method(**step_params)
                
                # Add to operations and results
                operations.append(f"Used {tool_name} with parameters: {step_params}")
                results.append(result)
            except Exception as e:
                return f"Error executing tool '{tool_name}': {str(e)}"
        
        # Format the final result
        final_result = results[-1]  # Use the result of the last step
        
        if isinstance(final_result, (pd.DataFrame, pd.Series)):
            formatted_result = self.pandas_tools.format_output(final_result)
        elif isinstance(final_result, tuple) and len(final_result) == 2 and isinstance(final_result[1], plt.Figure):
            # Handle the case where the result is a tuple of (dataframe, figure)
            df, fig = final_result
            # Save the figure to a temporary file
            fig_path = f"temp_figure_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            formatted_result = f"Figure saved to {fig_path}\n\n{self.pandas_tools.format_output(df)}"
        else:
            formatted_result = str(final_result)
        
        # Generate an explanation
        explanation = self.llm.explain_result(formatted_result, query, operations)
        
        return explanation
    
    def _process_dynamic_plan(self, query: str) -> str:
        """
        Process a request that requires a dynamic plan with multiple tools.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Response to the user.
        """
        # Generate a dynamic plan
        plan = self.llm.generate_dynamic_plan(query, self.dataframe_info)
        
        # Execute the plan
        operations = []
        results = []
        
        for step in plan:
            # Extract the tool name and parameters
            if "operation" in step:
                tool_name = step["operation"]
            else:
                return f"Missing operation in step: {step}"
            
            # Check if the tool exists
            if tool_name not in self.available_tools and tool_name != "custom":
                return f"Tool '{tool_name}' not found."
            
            # Extract parameters
            params = step.get("parameters", {})
            
            # Execute the tool
            try:
                if tool_name == "custom":
                    # Handle custom operations
                    custom_query = params.get("query", "")
                    # Process the custom query recursively
                    result = self.process_request(custom_query)
                else:
                    tool_method = self.available_tools[tool_name]["method"]
                    result = tool_method(**params)
                
                # Add to operations and results
                operations.append(f"Used {tool_name} with parameters: {params}")
                results.append(result)
            except Exception as e:
                return f"Error executing tool '{tool_name}': {str(e)}"
        
        # Format the final result
        final_result = results[-1]  # Use the result of the last step
        
        if isinstance(final_result, (pd.DataFrame, pd.Series)):
            formatted_result = self.pandas_tools.format_output(final_result)
        elif isinstance(final_result, tuple) and len(final_result) == 2 and isinstance(final_result[1], plt.Figure):
            # Handle the case where the result is a tuple of (dataframe, figure)
            df, fig = final_result
            # Save the figure to a temporary file
            fig_path = f"temp_figure_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            fig.savefig(fig_path)
            plt.close(fig)
            formatted_result = f"Figure saved to {fig_path}\n\n{self.pandas_tools.format_output(df)}"
        else:
            formatted_result = str(final_result)
        
        # Generate an explanation
        explanation = self.llm.explain_result(formatted_result, query, operations)
        
        return explanation
    
    def _process_fallback(self, query: str) -> str:
        """
        Process a request that cannot be handled by the available tools.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Response to the user.
        """
        # Format the prompt
        prompt = f"""
        The following user query cannot be handled by the available tools:
        
        User query: {query}
        
        Available tools:
        {', '.join(self.available_tools.keys())}
        
        Dataframe information:
        {self.dataframe_info}
        
        Please provide a helpful response explaining why the query cannot be handled and suggest alternatives.
        """
        
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains why queries cannot be handled and suggests alternatives."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.get_llm_response(messages)
        
        return response
    
    def run_interactive(self):
        """
        Run the agent in interactive mode.
        """
        print("Welcome to the Table Processing Agent!")
        print("Type 'exit' or 'quit' to exit.")
        print("Type 'load <file_path>' to load a dataframe.")
        
        while True:
            try:
                # Get user input
                user_input = input("\nEnter your request: ")
                
                # Check if the user wants to exit
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Process the request
                response = self.process_request(user_input)
                
                # Print the response
                print("\nResponse:")
                print(response)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    agent = TableAgent()
    agent.run_interactive() 