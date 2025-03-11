"""
Predefined pandas operations for table processing.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tabulate import tabulate
import re
from typing import List, Dict, Any, Union, Tuple, Optional

class PandasTools:
    """
    A collection of pandas operations for table processing.
    """
    def __init__(self, df=None):
        """
        Initialize with an optional dataframe.
        
        Args:
            df (pd.DataFrame, optional): Initial dataframe. Defaults to None.
        """
        self.df = df
        self.history = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            file_path (str): Path to the file.
            
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                self.df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self._add_to_history(f"Loaded data from {file_path}")
            return self.df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _add_to_history(self, operation: str):
        """
        Add an operation to the history.
        
        Args:
            operation (str): Description of the operation.
        """
        self.history.append({
            'timestamp': datetime.now(),
            'operation': operation
        })
    
    # Simple operations (single pandas function call)
    
    def get_column_names(self) -> List[str]:
        """
        Get the column names of the dataframe.
        
        Returns:
            List[str]: List of column names.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        self._add_to_history("Retrieved column names")
        return list(self.df.columns)
    
    def get_dataframe_info(self) -> str:
        """
        Get information about the dataframe.
        
        Returns:
            str: String representation of dataframe info.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        import io
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        self._add_to_history("Retrieved dataframe info")
        return info_str
    
    def get_data_head(self, n: int = 5) -> pd.DataFrame:
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int, optional): Number of rows. Defaults to 5.
            
        Returns:
            pd.DataFrame: First n rows of the dataframe.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        self._add_to_history(f"Retrieved first {n} rows")
        return self.df.head(n)
    
    def get_data_tail(self, n: int = 5) -> pd.DataFrame:
        """
        Get the last n rows of the dataframe.
        
        Args:
            n (int, optional): Number of rows. Defaults to 5.
            
        Returns:
            pd.DataFrame: Last n rows of the dataframe.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        self._add_to_history(f"Retrieved last {n} rows")
        return self.df.tail(n)
    
    def get_column_data(self, column_name: str) -> pd.Series:
        """
        Get data from a specific column.
        
        Args:
            column_name (str): Name of the column.
            
        Returns:
            pd.Series: Column data.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        self._add_to_history(f"Retrieved data from column '{column_name}'")
        return self.df[column_name]
    
    def get_row_data(self, row_index: int) -> pd.Series:
        """
        Get data from a specific row.
        
        Args:
            row_index (int): Index of the row.
            
        Returns:
            pd.Series: Row data.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if row_index < 0 or row_index >= len(self.df):
            raise ValueError(f"Row index {row_index} out of bounds")
        
        self._add_to_history(f"Retrieved data from row {row_index}")
        return self.df.iloc[row_index]
    
    def calculate_column_mean(self, column_name: str) -> float:
        """
        Calculate the mean of a column.
        
        Args:
            column_name (str): Name of the column.
            
        Returns:
            float: Mean value.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        self._add_to_history(f"Calculated mean of column '{column_name}'")
        return self.df[column_name].mean()
    
    def calculate_column_sum(self, column_name: str) -> float:
        """
        Calculate the sum of a column.
        
        Args:
            column_name (str): Name of the column.
            
        Returns:
            float: Sum value.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        self._add_to_history(f"Calculated sum of column '{column_name}'")
        return self.df[column_name].sum()
    
    def calculate_column_median(self, column_name: str) -> float:
        """
        Calculate the median of a column.
        
        Args:
            column_name (str): Name of the column.
            
        Returns:
            float: Median value.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        self._add_to_history(f"Calculated median of column '{column_name}'")
        return self.df[column_name].median()
    
    def calculate_column_std(self, column_name: str) -> float:
        """
        Calculate the standard deviation of a column.
        
        Args:
            column_name (str): Name of the column.
            
        Returns:
            float: Standard deviation value.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        self._add_to_history(f"Calculated standard deviation of column '{column_name}'")
        return self.df[column_name].std()
    
    def group_by_column(self, group_column: str, agg_column: str = None, agg_func: str = 'mean') -> pd.DataFrame:
        """
        Group data by a column and apply an aggregation function.
        
        Args:
            group_column (str): Column to group by.
            agg_column (str, optional): Column to aggregate. Defaults to None.
            agg_func (str, optional): Aggregation function. Defaults to 'mean'.
            
        Returns:
            pd.DataFrame: Grouped dataframe.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if group_column not in self.df.columns:
            raise ValueError(f"Column '{group_column}' not found")
        
        if agg_column is not None and agg_column not in self.df.columns:
            raise ValueError(f"Column '{agg_column}' not found")
        
        if agg_column is None:
            result = self.df.groupby(group_column).size().reset_index(name='count')
            self._add_to_history(f"Grouped data by '{group_column}' and counted occurrences")
        else:
            result = self.df.groupby(group_column)[agg_column].agg(agg_func).reset_index()
            self._add_to_history(f"Grouped data by '{group_column}' and calculated {agg_func} of '{agg_column}'")
        
        return result
    
    def filter_data(self, column: str, condition: str, value: Any) -> pd.DataFrame:
        """
        Filter data based on a condition.
        
        Args:
            column (str): Column to filter on.
            condition (str): Condition to apply ('==', '!=', '>', '<', '>=', '<=', 'contains', 'startswith', 'endswith').
            value (Any): Value to compare against.
            
        Returns:
            pd.DataFrame: Filtered dataframe.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        if condition == '==':
            result = self.df[self.df[column] == value]
        elif condition == '!=':
            result = self.df[self.df[column] != value]
        elif condition == '>':
            result = self.df[self.df[column] > value]
        elif condition == '<':
            result = self.df[self.df[column] < value]
        elif condition == '>=':
            result = self.df[self.df[column] >= value]
        elif condition == '<=':
            result = self.df[self.df[column] <= value]
        elif condition == 'contains':
            result = self.df[self.df[column].astype(str).str.contains(str(value), na=False)]
        elif condition == 'startswith':
            result = self.df[self.df[column].astype(str).str.startswith(str(value), na=False)]
        elif condition == 'endswith':
            result = self.df[self.df[column].astype(str).str.endswith(str(value), na=False)]
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        
        self._add_to_history(f"Filtered data where '{column}' {condition} '{value}'")
        return result
    
    # Complex operations (multiple pandas function calls)
    
    def generate_annual_report(self, date_column: str, value_column: str) -> pd.DataFrame:
        """
        Generate an annual report by aggregating data by year.
        
        Args:
            date_column (str): Column containing dates.
            value_column (str): Column containing values to aggregate.
            
        Returns:
            pd.DataFrame: Annual report.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found")
        
        if value_column not in self.df.columns:
            raise ValueError(f"Column '{value_column}' not found")
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column])
            except:
                raise ValueError(f"Could not convert '{date_column}' to datetime")
        
        # Extract year and group by it
        self.df['Year'] = self.df[date_column].dt.year
        annual_report = self.df.groupby('Year')[value_column].agg(['sum', 'mean', 'min', 'max', 'count']).reset_index()
        
        self._add_to_history(f"Generated annual report for '{value_column}' based on '{date_column}'")
        return annual_report
    
    def generate_monthly_report(self, date_column: str, value_column: str, year: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a monthly report by aggregating data by month.
        
        Args:
            date_column (str): Column containing dates.
            value_column (str): Column containing values to aggregate.
            year (int, optional): Specific year to filter for. Defaults to None.
            
        Returns:
            pd.DataFrame: Monthly report.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found")
        
        if value_column not in self.df.columns:
            raise ValueError(f"Column '{value_column}' not found")
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column])
            except:
                raise ValueError(f"Could not convert '{date_column}' to datetime")
        
        # Filter by year if specified
        if year is not None:
            filtered_df = self.df[self.df[date_column].dt.year == year].copy()
        else:
            filtered_df = self.df.copy()
        
        # Extract year and month
        filtered_df['Year'] = filtered_df[date_column].dt.year
        filtered_df['Month'] = filtered_df[date_column].dt.month
        
        # Group by year and month
        monthly_report = filtered_df.groupby(['Year', 'Month'])[value_column].agg(['sum', 'mean', 'min', 'max', 'count']).reset_index()
        
        # Add month names
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        monthly_report['Month_Name'] = monthly_report['Month'].map(month_names)
        
        # Reorder columns
        monthly_report = monthly_report[['Year', 'Month', 'Month_Name', 'sum', 'mean', 'min', 'max', 'count']]
        
        if year is not None:
            self._add_to_history(f"Generated monthly report for '{value_column}' in year {year} based on '{date_column}'")
        else:
            self._add_to_history(f"Generated monthly report for '{value_column}' based on '{date_column}'")
        
        return monthly_report
    
    def analyze_trend(self, date_column: str, value_column: str, freq: str = 'M') -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze the trend of a value over time.
        
        Args:
            date_column (str): Column containing dates.
            value_column (str): Column containing values to analyze.
            freq (str, optional): Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly, 'Y' for yearly). Defaults to 'M'.
            
        Returns:
            Tuple[pd.DataFrame, plt.Figure]: Trend dataframe and plot.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if date_column not in self.df.columns:
            raise ValueError(f"Column '{date_column}' not found")
        
        if value_column not in self.df.columns:
            raise ValueError(f"Column '{value_column}' not found")
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            try:
                self.df[date_column] = pd.to_datetime(self.df[date_column])
            except:
                raise ValueError(f"Could not convert '{date_column}' to datetime")
        
        # Set date as index
        df_trend = self.df.copy()
        df_trend.set_index(date_column, inplace=True)
        
        # Resample data
        trend_data = df_trend[value_column].resample(freq).mean()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        trend_data.plot(ax=ax)
        ax.set_title(f'Trend of {value_column} over time')
        ax.set_ylabel(value_column)
        ax.grid(True)
        
        # Reset index for the dataframe
        trend_df = trend_data.reset_index()
        
        freq_map = {
            'D': 'daily',
            'W': 'weekly',
            'M': 'monthly',
            'Q': 'quarterly',
            'Y': 'yearly'
        }
        freq_name = freq_map.get(freq, freq)
        
        self._add_to_history(f"Analyzed {freq_name} trend of '{value_column}' based on '{date_column}'")
        return trend_df, fig
    
    def correlation_analysis(self, columns: List[str] = None) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Analyze correlations between columns.
        
        Args:
            columns (List[str], optional): List of columns to analyze. Defaults to None (all numeric columns).
            
        Returns:
            Tuple[pd.DataFrame, plt.Figure]: Correlation matrix and heatmap.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        # Select numeric columns if not specified
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            columns = list(numeric_cols)
        else:
            # Verify all columns exist and are numeric
            for col in columns:
                if col not in self.df.columns:
                    raise ValueError(f"Column '{col}' not found")
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise ValueError(f"Column '{col}' is not numeric")
        
        if len(columns) < 2:
            raise ValueError("Need at least two numeric columns for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = self.df[columns].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        
        self._add_to_history(f"Analyzed correlations between {len(columns)} columns")
        return corr_matrix, fig
    
    # Auxiliary functions
    
    def convert_date_references(self, date_reference: str) -> datetime:
        """
        Convert relative date references to actual dates.
        
        Args:
            date_reference (str): Date reference (e.g., 'today', 'yesterday', 'last week').
            
        Returns:
            datetime: Converted date.
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if date_reference.lower() == 'today':
            return today
        elif date_reference.lower() == 'yesterday':
            return today - timedelta(days=1)
        elif date_reference.lower() == 'tomorrow':
            return today + timedelta(days=1)
        elif 'last week' in date_reference.lower():
            return today - timedelta(weeks=1)
        elif 'last month' in date_reference.lower():
            # Approximate a month as 30 days
            return today - timedelta(days=30)
        elif 'last year' in date_reference.lower():
            # Approximate a year as 365 days
            return today - timedelta(days=365)
        elif 'next week' in date_reference.lower():
            return today + timedelta(weeks=1)
        elif 'next month' in date_reference.lower():
            # Approximate a month as 30 days
            return today + timedelta(days=30)
        elif 'next year' in date_reference.lower():
            # Approximate a year as 365 days
            return today + timedelta(days=365)
        else:
            # Try to parse as a date string
            try:
                return pd.to_datetime(date_reference)
            except:
                raise ValueError(f"Could not parse date reference: {date_reference}")
    
    def string_operations(self, column: str, operation: str, pattern: str = None) -> pd.Series:
        """
        Perform string operations on a column.
        
        Args:
            column (str): Column to operate on.
            operation (str): Operation to perform ('upper', 'lower', 'title', 'strip', 'replace', 'extract').
            pattern (str, optional): Pattern for replace or extract operations. Defaults to None.
            
        Returns:
            pd.Series: Processed column.
        """
        if self.df is None:
            raise ValueError("No dataframe loaded")
        
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        # Convert column to string if it's not already
        series = self.df[column].astype(str)
        
        if operation == 'upper':
            result = series.str.upper()
            self._add_to_history(f"Converted '{column}' to uppercase")
        elif operation == 'lower':
            result = series.str.lower()
            self._add_to_history(f"Converted '{column}' to lowercase")
        elif operation == 'title':
            result = series.str.title()
            self._add_to_history(f"Converted '{column}' to title case")
        elif operation == 'strip':
            result = series.str.strip()
            self._add_to_history(f"Stripped whitespace from '{column}'")
        elif operation == 'replace' and pattern is not None:
            # Pattern should be in the format 'old_value|new_value'
            try:
                old_val, new_val = pattern.split('|')
                result = series.str.replace(old_val, new_val)
                self._add_to_history(f"Replaced '{old_val}' with '{new_val}' in '{column}'")
            except:
                raise ValueError("Pattern for replace should be in the format 'old_value|new_value'")
        elif operation == 'extract' and pattern is not None:
            # Extract using regex pattern
            try:
                result = series.str.extract(pattern, expand=False)
                self._add_to_history(f"Extracted pattern '{pattern}' from '{column}'")
            except:
                raise ValueError(f"Invalid regex pattern: {pattern}")
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return result
    
    def format_output(self, data: Union[pd.DataFrame, pd.Series, float, int, str], format_type: str = 'table') -> str:
        """
        Format output data for display.
        
        Args:
            data: Data to format.
            format_type (str, optional): Format type ('table', 'json', 'csv', 'markdown'). Defaults to 'table'.
            
        Returns:
            str: Formatted output.
        """
        if isinstance(data, (float, int)):
            return str(data)
        elif isinstance(data, str):
            return data
        elif isinstance(data, pd.Series):
            data = data.to_frame()
        
        if format_type == 'table':
            return tabulate(data, headers='keys', tablefmt='psql', showindex=True)
        elif format_type == 'json':
            return data.to_json(orient='records', indent=2)
        elif format_type == 'csv':
            return data.to_csv(index=False)
        elif format_type == 'markdown':
            return data.to_markdown(index=False)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of operations.
        
        Returns:
            List[Dict[str, Any]]: List of operations.
        """
        return self.history 