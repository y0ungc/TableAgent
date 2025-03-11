"""
Setup script for the table processing agent.
"""
from setuptools import setup, find_packages

setup(
    name="table_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "datetime>=5.1",
        "langchain>=0.0.267",
        "tabulate>=0.9.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A table processing agent that uses LLM and pandas to automatically process tables based on user requests.",
    keywords="pandas, llm, table, processing, agent",
    url="https://github.com/yourusername/table_agent",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 