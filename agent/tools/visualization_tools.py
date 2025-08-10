import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
from sklearn.linear_model import LinearRegression
import logging
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Tool for generating visualizations and charts"""
    
    def __init__(self):
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set figure parameters for smaller file sizes
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['figure.figsize'] = [10, 6]
    
    def create_visualization(self, question: str, data_context: Dict[str, Any]) -> Optional[str]:
        """
        Create visualization based on question and return as base64 encoded image
        
        Args:
            question: Question/request for visualization
            data_context: Available data
            
        Returns:
            Base64 encoded image as data URI or None if failed
        """
        try:
            question_lower = question.lower()
            if 'scatterplot' in question_lower or ('plot' in question_lower and 'rank' in question_lower and 'peak' in question_lower):
                return self._create_scatterplot(question, data_context)
            elif 'plot' in question_lower and ('year' in question_lower or 'delay' in question_lower):
                return self._create_time_series_plot(question, data_context)
            elif any(word in question_lower for word in ['histogram', 'bar', 'chart']):
                return self._create_bar_chart(question, data_context)
            else:
                return self._create_scatterplot(question, data_context)
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return None
    
    def _create_scatterplot(self, question: str, data_context: Dict[str, Any]) -> Optional[str]:
        """Create scatterplot with regression line"""
        try:
            # Find the main dataframe
            main_data = self._find_main_dataframe(data_context)
            if main_data is None:
                return None
            
            # Determine axes from question
            x_col, y_col = self._extract_axes_from_question(question, main_data)
            
            if not x_col or not y_col:
                return None
            
            # Clean the data
            x_data = pd.to_numeric(main_data[x_col], errors='coerce')
            y_data = pd.to_numeric(main_data[y_col], errors='coerce')
            
            # Remove NaN values
            valid_mask = ~(x_data.isna() | y_data.isna())
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]
            
            if len(x_clean) < 2:
                return None
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatterplot
            ax.scatter(x_clean, y_clean, alpha=0.6, s=50)
            
            # Add regression line
            if len(x_clean) > 1:
                X = x_clean.values.reshape(-1, 1)
                y = y_clean.values
                
                reg = LinearRegression()
                reg.fit(X, y)
                
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                y_line = reg.predict(x_line.reshape(-1, 1))
                
                # Red dotted regression line as specified
                ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'Regression Line')
            
            # Formatting
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'Scatterplot: {x_col} vs {y_col}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Convert to base64
            return self._fig_to_base64(fig, max_size_kb=100)
            
        except Exception as e:
            logger.error(f"Failed to create scatterplot: {e}")
            return None
    
    def _create_time_series_plot(self, question: str, data_context: Dict[str, Any]) -> Optional[str]:
        """Create time series plot"""
        try:
            main_data = self._find_main_dataframe(data_context)
            if main_data is None:
                return None
            
            # Look for year and delay data
            year_col = self._find_year_column(main_data)
            
            # Calculate delays if needed
            delay_data = None
            if 'delay' in question.lower():
                reg_col = self._find_column_by_name(main_data, ['registration', 'date_of_registration'])
                decision_col = self._find_column_by_name(main_data, ['decision', 'decision_date'])
                
                if reg_col and decision_col:
                    delay_data = self._calculate_delays(main_data, reg_col, decision_col)
            
            if not year_col or delay_data is None:
                return None
            
            # Group by year and calculate mean delay
            year_data = main_data[year_col]
            df_plot = pd.DataFrame({'year': year_data, 'delay': delay_data})
            df_plot = df_plot.dropna()
            
            if df_plot.empty:
                return None
            
            yearly_delays = df_plot.groupby('year')['delay'].mean()
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Scatterplot
            ax.scatter(yearly_delays.index, yearly_delays.values, alpha=0.7, s=50)
            
            # Add regression line
            if len(yearly_delays) > 1:
                X = yearly_delays.index.values.reshape(-1, 1)
                y = yearly_delays.values
                
                reg = LinearRegression()
                reg.fit(X, y)
                
                x_line = np.linspace(yearly_delays.index.min(), yearly_delays.index.max(), 100)
                y_line = reg.predict(x_line.reshape(-1, 1))
                
                ax.plot(x_line, y_line, 'r--', linewidth=2, label='Regression Line')
            
            # Formatting
            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel('Days of Delay', fontsize=12)
            ax.set_title('Year vs Days of Delay', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            return self._fig_to_base64(fig, max_size_kb=100)
            
        except Exception as e:
            logger.error(f"Failed to create time series plot: {e}")
            return None
    
    def _create_bar_chart(self, question: str, data_context: Dict[str, Any]) -> Optional[str]:
        """Create bar chart"""
        try:
            main_data = self._find_main_dataframe(data_context)
            if main_data is None:
                return None
            
            # Find categorical and numeric columns
            categorical_cols = main_data.select_dtypes(include=['object']).columns
            numeric_cols = main_data.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                return None
            
            # Use first categorical and numeric columns
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Group and aggregate
            grouped = main_data.groupby(cat_col)[num_col].mean().head(10)  # Top 10
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(grouped)), grouped.values)
            
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_ylabel(num_col)
            ax.set_title(f'{num_col} by {cat_col}')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, max_size_kb=100)
            
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
            return None
    
    def _extract_axes_from_question(self, question: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Extract x and y axis column names from question"""
        question_lower = question.lower()
        
        # Look for specific column mentions
        if 'rank' in question_lower and 'peak' in question_lower:
            rank_col = self._find_column_by_name(df, ['rank'])
            peak_col = self._find_column_by_name(df, ['peak'])
            return rank_col, peak_col
        
        # Default to first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            return numeric_cols[0], numeric_cols[1]
        
        return None, None
    
    def _find_main_dataframe(self, data_context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Find the main DataFrame in the data context"""
        for key, value in data_context.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                return value
            elif isinstance(value, dict):
                if 'movies_data' in value and isinstance(value['movies_data'], list):
                    return pd.DataFrame(value['movies_data'])
                elif 'tables' in value and isinstance(value['tables'], list):
                    for table in value['tables']:
                        if 'data' in table:
                            df = pd.DataFrame(table['data'])
                            if not df.empty:
                                return df
        
        # Check for SQL results
        for key, value in data_context.items():
            if 'sql_result' in key and isinstance(value, pd.DataFrame):
                return value
        
        return None
    
    def _find_year_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find column containing year data"""
        return self._find_column_by_name(df, ['year', 'release_year', 'date'])
    
    def _find_column_by_name(self, df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
        """Find column that matches any of the keywords"""
        columns_lower = [col.lower() for col in df.columns]
        
        for keyword in keywords:
            for i, col_lower in enumerate(columns_lower):
                if keyword in col_lower:
                    return df.columns[i]
        
        return None
    
    def _calculate_delays(self, df: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
        """Calculate delays between two date columns"""
        try:
            start_dates = pd.to_datetime(df[start_col], errors='coerce')
            end_dates = pd.to_datetime(df[end_col], errors='coerce')
            
            delays = (end_dates - start_dates).dt.days
            return delays
        except Exception as e:
            logger.error(f"Failed to calculate delays: {e}")
            return pd.Series([np.nan] * len(df))
    
    def _fig_to_base64(self, fig, max_size_kb: int = 100) -> str:
        """Convert matplotlib figure to base64 string with size constraint"""
        try:
            # Try reducing figure size progressively to meet size cap
            sizes = [(10,6), (8,5), (7,4.5), (6,4), (5,3.5)]
            last_image_data = None
            for w, h in sizes:
                buf = io.BytesIO()
                fig.set_size_inches(w, h)
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                image_data = buf.read()
                buf.close()
                size_kb = len(image_data) / 1024
                logger.info(f"Image size: {size_kb:.1f} KB at {w}x{h}")
                if size_kb <= max_size_kb:
                    encoded = base64.b64encode(image_data).decode('utf-8')
                    plt.close(fig)
                    return f"data:image/png;base64,{encoded}"
                last_image_data = image_data
            # Fallback to last attempt even if slightly over
            encoded = base64.b64encode(last_image_data).decode('utf-8') if last_image_data else ''
            plt.close(fig)
            return f"data:image/png;base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to convert figure to base64: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def create_custom_plot(self, plot_type: str, data: pd.DataFrame, 
                          x_col: str, y_col: str = None, 
                          title: str = None) -> Optional[str]:
        """Create custom plot with specified parameters"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == 'scatter':
                if y_col:
                    ax.scatter(data[x_col], data[y_col], alpha=0.6)
                    ax.set_ylabel(y_col)
            elif plot_type == 'line':
                if y_col:
                    ax.plot(data[x_col], data[y_col], marker='o')
                    ax.set_ylabel(y_col)
            elif plot_type == 'bar':
                value_counts = data[x_col].value_counts().head(10)
                ax.bar(range(len(value_counts)), value_counts.values)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            elif plot_type == 'histogram':
                ax.hist(data[x_col].dropna(), bins=30, alpha=0.7)
            
            ax.set_xlabel(x_col)
            if title:
                ax.set_title(title)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Failed to create custom plot: {e}")
            return None