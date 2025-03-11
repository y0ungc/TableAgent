"""
Example script to demonstrate how to use the table processing agent.
"""
import os
import sys
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the table agent
from main import TableAgent

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Main function to demonstrate the table processing agent.
    """
    # Create a table agent
    agent = TableAgent()
    
    # Load the sample data
    print("Loading sample data...")
    response = agent.load_data("sample_data.csv")
    print(response)
    print()
    
    # Example 1: Simple request
    print("Example 1: Simple request")
    request = "What are the column names in this table?"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    # Example 2: Calculate average
    print("Example 2: Calculate average")
    request = "What is the average sales value?"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    # Example 3: Group by
    print("Example 3: Group by")
    request = "Group the data by category and show the total sales for each category"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    # Example 4: Complex request (predefined chain)
    print("Example 4: Complex request (predefined chain)")
    request = "Generate a monthly report of sales for 2023"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    # Example 5: Dynamic plan
    print("Example 5: Dynamic plan")
    request = "Show me the trend of sales for electronics products over time"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    # Example 6: Natural language with context
    print("Example 6: Natural language with context")
    request = "Which region had the highest sales in the last quarter of the year?"
    print(f"Request: {request}")
    response = agent.process_request(request)
    print(f"Response: {response}")
    print()
    
    print("Examples completed. You can now run the interactive mode with 'python main.py'")

if __name__ == "__main__":
    main() 