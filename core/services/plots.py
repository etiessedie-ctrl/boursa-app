"""
Visualization service for generating charts and plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import plotly, but make it optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Interactive charts will be disabled.")
import json
from io import BytesIO
import base64
from typing import Dict, List, Any, Optional
from typing import Dict, List, Any, Optional
import os
from django.conf import settings


class VisualizationService:
    """Service for generating various types of visualizations"""

    @staticmethod
    def create_histogram(data: pd.Series, title: str = "Histogram", bins: int = 30) -> str:
        """Create a histogram plot"""
        plt.figure(figsize=(10, 6))
        plt.hist(data.dropna(), bins=bins, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel(data.name or 'Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_boxplot(data: pd.DataFrame, title: str = "Box Plot") -> str:
        """Create a box plot for multiple columns"""
        plt.figure(figsize=(12, 8))
        numeric_columns = data.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10 columns
        if len(numeric_columns) > 0:
            data[numeric_columns].boxplot()
            plt.title(title)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No numeric columns found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

    @staticmethod
    def create_qqplot(data: pd.Series, title: str = "Q-Q Plot") -> str:
        """Create a Q-Q plot for normality testing"""
        import scipy.stats as stats
        
        plt.figure(figsize=(10, 6))
        data_clean = data.dropna()
        if len(data_clean) > 0:
            stats.probplot(data_clean, dist="norm", plot=plt)
            plt.title(title)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No data available for Q-Q plot', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"
        """Create a scatter plot"""
        plt.figure(figsize=(10, 6))
        if x_col in data.columns and y_col in data.columns:
            plt.scatter(data[x_col], data[y_col], alpha=0.6)
            plt.title(title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Columns {x_col} or {y_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Scatter Plot") -> str:
        """Create a scatter plot"""
        plt.figure(figsize=(10, 6))
        if x_col in data.columns and y_col in data.columns:
            plt.scatter(data[x_col], data[y_col], alpha=0.6)
            plt.title(title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Columns {x_col} or {y_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

    @staticmethod
    def create_scatter_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Scatter Plot") -> str:
        """Create a scatter plot"""
        plt.figure(figsize=(10, 6))
        if x_col in data.columns and y_col in data.columns:
            plt.scatter(data[x_col], data[y_col], alpha=0.6)
            plt.title(title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Columns {x_col} or {y_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_line_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Line Plot") -> str:
        """Create a line plot"""
        plt.figure(figsize=(10, 6))
        if x_col in data.columns and y_col in data.columns:
            plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', alpha=0.7)
            plt.title(title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Columns {x_col} or {y_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_bar_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str = "Bar Plot") -> str:
        """Create a bar plot"""
        plt.figure(figsize=(10, 6))
        if x_col in data.columns and y_col in data.columns:
            plt.bar(data[x_col], data[y_col], alpha=0.7)
            plt.title(title)
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, f'Columns {x_col} or {y_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_correlation_heatmap(data: pd.DataFrame, title: str = "Correlation Heatmap") -> str:
        plt.figure(figsize=(12, 10))
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title(title)
        else:
            plt.text(0.5, 0.5, 'No numeric columns found for correlation', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_time_series_plot(data: pd.DataFrame, date_col: str, value_col: str, title: str = "Time Series") -> str:
        """Create a time series plot"""
        plt.figure(figsize=(14, 7))
        if date_col in data.columns and value_col in data.columns:
            # Try to convert date column to datetime
            try:
                dates = pd.to_datetime(data[date_col])
                plt.plot(dates, data[value_col], marker='o', linestyle='-', alpha=0.7)
                plt.title(title)
                plt.xlabel('Date')
                plt.ylabel(value_col)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
            except:
                plt.text(0.5, 0.5, f'Could not parse dates from column {date_col}', ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, f'Columns {date_col} or {value_col} not found', ha='center', va='center', transform=plt.gca().transAxes)

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    @staticmethod
    def create_interactive_plotly_chart(data: pd.DataFrame, chart_type: str = 'scatter',
                                      x_col: str = None, y_col: str = None,
                                      title: str = "Interactive Chart") -> Dict[str, Any]:
        """Create an interactive Plotly chart"""
        if not PLOTLY_AVAILABLE:
            return {'error': 'Plotly not installed. Interactive charts are not available.'}

        try:
            if chart_type == 'scatter' and x_col and y_col:
                fig = px.scatter(data, x=x_col, y=y_col, title=title)
            elif chart_type == 'line' and x_col and y_col:
                fig = px.line(data, x=x_col, y=y_col, title=title)
            elif chart_type == 'bar':
                numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]
                if len(numeric_cols) > 0:
                    fig = px.bar(data, x=data.index[:50], y=numeric_cols[0], title=title)
                else:
                    return {'error': 'No numeric columns found'}
            elif chart_type == 'histogram':
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.histogram(data, x=numeric_cols[0], title=title)
                else:
                    return {'error': 'No numeric columns found'}
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}

            # Convert to JSON for frontend
            chart_json = json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
            return {
                'data': chart_json['data'],
                'layout': chart_json['layout']
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive data summary"""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Numeric columns summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = data[numeric_cols].describe().to_dict()

        # Categorical columns summary
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols[:5]:  # Limit to 5 columns
                value_counts = data[col].value_counts().head(10)  # Top 10 values
                cat_summary[col] = {
                    'unique_values': data[col].nunique(),
                    'top_values': value_counts.to_dict()
                }
            summary['categorical_summary'] = cat_summary

        return summary

    @staticmethod
    def generate_all_visualizations(data: pd.DataFrame) -> Dict[str, Any]:
        """Generate all available visualizations for the dataset"""
        visualizations = {}

        if data.empty:
            return {'error': 'Dataset is empty'}

        try:
            # Basic plots
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                # Histogram for first numeric column
                visualizations['histogram'] = VisualizationService.create_histogram(
                    data[numeric_cols[0]], f"Distribution of {numeric_cols[0]}"
                )

                # Box plot
                visualizations['boxplot'] = VisualizationService.create_boxplot(
                    data, "Box Plot of Numeric Columns"
                )

                # Correlation heatmap
                visualizations['correlation'] = VisualizationService.create_correlation_heatmap(
                    data, "Correlation Matrix"
                )

            # Scatter plot if we have at least 2 numeric columns
            if len(numeric_cols) >= 2:
                visualizations['scatter'] = VisualizationService.create_scatter_plot(
                    data, numeric_cols[0], numeric_cols[1],
                    f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}"
                )

            # Time series if we can detect date columns
            date_cols = []
            for col in data.columns:
                try:
                    pd.to_datetime(data[col].head())
                    date_cols.append(col)
                except:
                    continue

            if date_cols and numeric_cols.any():
                visualizations['timeseries'] = VisualizationService.create_time_series_plot(
                    data, date_cols[0], numeric_cols[0], f"Time Series: {numeric_cols[0]}"
                )

            # Interactive charts (only if plotly is available)
            if len(numeric_cols) > 0 and PLOTLY_AVAILABLE:
                visualizations['interactive_histogram'] = VisualizationService.create_interactive_plotly_chart(
                    data, 'histogram', title="Interactive Histogram"
                )

            # Data summary
            visualizations['summary'] = VisualizationService.get_data_summary(data)

        except Exception as e:
            visualizations['error'] = str(e)

        return visualizations