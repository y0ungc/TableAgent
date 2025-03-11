# Table Processing Agent

This project implements an intelligent agent that uses LLM and pandas to automatically process tables based on user requests.

## Features

- Simple operations: column names, row/column information, average calculations, grouping
- Complex operations: annual reports, trend analysis
- Auxiliary functions: date reformatting, string operations
- Context-aware request processing
- Dynamic tool selection based on request complexity

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the agent:
```bash
python main.py
```

## Usage

1. Load a CSV file:
```
Please load the file: path/to/your/file.csv
```

2. Ask questions about the data:
```
What is the average value of the 'Sales' column?
Show me the trend of revenue over the last 5 years
Generate a monthly report for 2023
```

3. The agent will process your request and return the results.

## Project Structure

- `main.py`: Entry point for the application
- `utils/pandas_tools.py`: Predefined pandas operations
- `utils/llm_integration.py`: LLM integration for understanding user requests
- `utils/config.py`: Configuration settings 