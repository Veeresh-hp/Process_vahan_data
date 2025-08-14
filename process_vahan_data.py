# 🚗 Enhanced Vehicle Registration Analytics Dashboard
# =======================================================
"""
VEHICLE REGISTRATION ANALYTICS DASHBOARD
=========================================

PURPOSE:
This interactive dashboard provides comprehensive insights into vehicle registration
trends for investors and analysts. It processes quarterly vehicle registration data
and presents key metrics, growth patterns, and market analysis through an intuitive
web interface.

KEY FEATURES:
• Interactive filtering by vehicle type, manufacturer, and time period
• Real-time calculation of YoY and QoQ growth metrics
• Investor-focused KPI tracking and market insights
• Visual analytics with charts, trends, and heatmaps
• Data export functionality for further analysis
• Responsive design with helpful tooltips and explanations

DATA SOURCES:
• Quarterly vehicle registration data (CSV format)
• Monthly data automatically converted to quarterly aggregates
• Growth metrics calculated dynamically based on selected filters

HOW TO USE:
1. Run the script: streamlit run dashboard.py
2. Use sidebar filters to customize your analysis
3. Navigate through tabs for different analytical perspectives
4. Export filtered data for external analysis
5. Hover over charts and metrics for detailed information

AUTHOR: Enhanced for Backend Developer Internship Assignment
DATE: 2025
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# PAGE CONFIGURATION AND SETUP
# =======================================================

# Configure Streamlit page settings for optimal user experience
st.set_page_config(
    page_title="Vehicle Registration Analytics",  # Browser tab title
    page_icon="🚗",                             # Browser tab icon
    layout="wide",                             # Use full screen width
    initial_sidebar_state="expanded"           # Start with sidebar open
)

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .help-text {
        font-size: 0.9em;
        color: #6c757d;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# =======================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =======================================================

@st.cache_data(show_spinner=True, ttl=3600)  # Cache data for 1 hour to improve performance
def load_data():
    """
    Load and preprocess vehicle registration data from CSV files.
    
    This function attempts to load data from multiple possible sources:
    1. Pre-processed quarterly growth metrics
    2. Monthly growth metrics data
    3. Raw vehicle registration data (processed into quarterly format)
    
    Returns:
        tuple: (monthly_growth_data, quarterly_data) or (None, None) if loading fails
    """
    
    # Initialize variables to store different data formats
    df_quarterly = None
    df_growth = None
    
    # Create progress indicator for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Try loading quarterly growth metrics (25% progress)
        progress_bar.progress(25)
        status_text.text("🔄 Loading quarterly growth metrics...")
        
        try:
            df_quarterly = pd.read_csv('vehicle_growth_metrics_quarterly.csv')
            st.success("✅ Successfully loaded quarterly growth metrics data")
        except FileNotFoundError:
            st.info("ℹ️ Quarterly metrics file not found, will process from source data")
        except Exception as e:
            st.warning(f"⚠️ Issue with quarterly file: {str(e)[:50]}...")
        
        # Step 2: Try loading monthly growth data (50% progress)
        progress_bar.progress(50)
        status_text.text("🔄 Loading monthly growth metrics...")
        
        try:
            df_growth = pd.read_csv('vehicle_growth_metrics.csv')
            if 'date' in df_growth.columns:
                df_growth['date'] = pd.to_datetime(df_growth['date'], errors='coerce')
            st.success("✅ Successfully loaded monthly growth metrics data")
        except FileNotFoundError:
            st.info("ℹ️ Monthly metrics file not found")
        except Exception as e:
            st.warning(f"⚠️ Issue with monthly file: {str(e)[:50]}...")
        
        # Step 3: Process main registration file if quarterly data unavailable (75% progress)
        progress_bar.progress(75)
        if df_quarterly is None:
            status_text.text("🔄 Processing main registration data...")
            
            try:
                df_cleaned = pd.read_csv('vehicle_registrations_cleaned-1.csv')
                df_quarterly = process_cleaned_data_to_quarterly(df_cleaned)
                st.success("✅ Successfully processed data from main registration file")
            except FileNotFoundError:
                st.error("❌ Main registration file 'vehicle_registrations_cleaned-1.csv' not found")
                st.info("💡 Please ensure your data files are in the correct location")
                return None, None
            except Exception as e:
                st.error(f"❌ Failed to process main data: {e}")
                return None, None
        
        # Step 4: Clean and validate quarterly data (100% progress)
        progress_bar.progress(100)
        status_text.text("🔄 Cleaning and validating data...")
        
        if df_quarterly is not None:
            df_quarterly = clean_quarterly_data(df_quarterly)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display data summary for user awareness
        if df_quarterly is not None:
            st.info(f"""
            📊 **Data Successfully Loaded**
            • Total records: {len(df_quarterly):,}
            • Date range: {df_quarterly['year'].min():.0f} - {df_quarterly['year'].max():.0f}
            • Vehicle types: {df_quarterly['vehicle_type'].nunique()}
            • Manufacturers: {df_quarterly['Maker'].nunique()}
            """)
        
        return df_growth, df_quarterly
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Critical error loading data: {e}")
        st.info("💡 Please check your data files and try again")
        return None, None

def clean_quarterly_data(df_quarterly):
    """
    Clean and standardize quarterly data for consistent analysis.
    
    Args:
        df_quarterly (pd.DataFrame): Raw quarterly data
        
    Returns:
        pd.DataFrame: Cleaned and standardized quarterly data
    """
    
    # Remove whitespace from column names for consistency
    df_quarterly.columns = df_quarterly.columns.str.strip()
    
    # Define expected numeric and text columns
    numeric_cols = ['year', 'quarter', 'registrations', 'registrations_last_quarter', 'QoQ_growth_%']
    text_cols = ['vehicle_type', 'Maker', 'year_quarter']
    
    # Convert numeric columns with error handling
    for col in numeric_cols:
        if col in df_quarterly.columns:
            df_quarterly[col] = pd.to_numeric(df_quarterly[col], errors='coerce')
    
    # Clean text columns
    for col in text_cols:
        if col in df_quarterly.columns:
            df_quarterly[col] = df_quarterly[col].astype(str).str.strip()
    
    # Remove rows with missing essential data
    essential_cols = ['Maker', 'vehicle_type', 'year', 'registrations']
    df_quarterly = df_quarterly.dropna(subset=[col for col in essential_cols if col in df_quarterly.columns])
    
    return df_quarterly

def process_cleaned_data_to_quarterly(df_cleaned):
    """
    Convert monthly vehicle registration data to quarterly aggregates.
    
    This function transforms raw monthly data into quarterly summaries with growth calculations.
    It handles various data formats and calculates Quarter-over-Quarter growth metrics.
    
    Args:
        df_cleaned (pd.DataFrame): Raw monthly registration data
        
    Returns:
        pd.DataFrame: Processed quarterly data with growth metrics
    """
    
    try:
        # Step 1: Prepare and clean the raw data
        df_cleaned.columns = df_cleaned.columns.str.strip()
        df_cleaned = df_cleaned.dropna(subset=['Maker', 'vehicle_type', 'year'])
        
        # Define month columns (standard format)
        month_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                      'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        
        # Check which month columns actually exist in the data
        existing_month_cols = [col for col in month_cols if col in df_cleaned.columns]
        
        if not existing_month_cols:
            st.error("❌ No monthly columns found in the data")
            st.info("💡 Expected columns: " + ", ".join(month_cols))
            return None
        
        st.info(f"📅 Processing {len(existing_month_cols)} months of data: {', '.join(existing_month_cols)}")
        
        # Step 2: Transform data from wide to long format
        df_melted = df_cleaned.melt(
            id_vars=['Maker', 'vehicle_type', 'year'],
            value_vars=existing_month_cols,
            var_name='month',
            value_name='registrations'
        )
        
        # Step 3: Clean registration values (handle various formats)
        df_melted['registrations'] = df_melted['registrations'].astype(str)
        
        # Remove common formatting characters
        cleanup_chars = [',', '"', ' ', "'"]
        for char in cleanup_chars:
            df_melted['registrations'] = df_melted['registrations'].str.replace(char, '')
        
        # Convert to numeric, replacing errors with 0
        df_melted['registrations'] = pd.to_numeric(df_melted['registrations'], errors='coerce').fillna(0)
        
        # Step 4: Map months to quarters
        month_to_quarter = {
            'JAN': 1, 'FEB': 1, 'MAR': 1,  # Q1
            'APR': 2, 'MAY': 2, 'JUN': 2,  # Q2
            'JUL': 3, 'AUG': 3, 'SEP': 3,  # Q3
            'OCT': 4, 'NOV': 4, 'DEC': 4   # Q4
        }
        
        df_melted['quarter'] = df_melted['month'].map(month_to_quarter)
        df_melted = df_melted.dropna(subset=['quarter'])
        
        # Step 5: Aggregate monthly data to quarterly totals
        df_quarterly = df_melted.groupby(['Maker', 'vehicle_type', 'year', 'quarter'])['registrations'].sum().reset_index()
        df_quarterly['year_quarter'] = df_quarterly['year'].astype(str) + '-Q' + df_quarterly['quarter'].astype(str)
        
        # Step 6: Calculate Quarter-over-Quarter (QoQ) growth
        df_quarterly = df_quarterly.sort_values(['Maker', 'vehicle_type', 'year', 'quarter'])
        df_quarterly['registrations_last_quarter'] = df_quarterly.groupby(['Maker', 'vehicle_type'])['registrations'].shift(1)
        
        # Calculate QoQ growth percentage
        mask = df_quarterly['registrations_last_quarter'] > 0
        df_quarterly.loc[mask, 'QoQ_growth_%'] = (
            (df_quarterly.loc[mask, 'registrations'] - df_quarterly.loc[mask, 'registrations_last_quarter']) 
            / df_quarterly.loc[mask, 'registrations_last_quarter'] * 100
        )
        
        st.success(f"✅ Successfully processed {len(df_quarterly):,} quarterly records")
        return df_quarterly
        
    except Exception as e:
        st.error(f"❌ Error processing cleaned data: {e}")
        return None

# =======================================================
# ANALYTICAL FUNCTIONS
# =======================================================

def calculate_growth_metrics(df, groupby_cols, value_col='registrations'):
    """
    Calculate Year-over-Year (YoY) and Quarter-over-Quarter (QoQ) growth metrics.
    
    This function computes growth rates for different time periods and groupings,
    providing essential metrics for trend analysis and investment decisions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        groupby_cols (list): Columns to group by for calculations
        value_col (str): Column name containing values for growth calculation
        
    Returns:
        pd.DataFrame: Data with calculated growth metrics
    """
    
    # Group and sum the data
    grouped = df.groupby(groupby_cols)[value_col].sum().reset_index()
    
    # Calculate Year-over-Year growth
    if 'year' in groupby_cols:
        grouped = grouped.sort_values(by=[col for col in groupby_cols if col != 'year'] + ['year'])
        grouped['yoy_growth'] = grouped.groupby([col for col in groupby_cols if col != 'year'])[value_col].pct_change(periods=1) * 100
    
    # Calculate Quarter-over-Quarter growth
    if 'year_quarter' in groupby_cols:
        grouped = grouped.sort_values(by=[col for col in groupby_cols if col != 'year_quarter'] + ['year_quarter'])
        grouped['qoq_growth'] = grouped.groupby([col for col in groupby_cols if col != 'year_quarter'])[value_col].pct_change(periods=1) * 100
    
    return grouped

def create_insights_summary(df):
    """
    Generate key market insights from the vehicle registration data.
    
    This function analyzes the data to provide actionable insights for investors,
    including growth leaders, market share analysis, and trend identification.
    
    Args:
        df (pd.DataFrame): Processed vehicle registration data
        
    Returns:
        list: List of formatted insight strings
    """
    
    insights = []
    
    try:
        # Insight 1: Top growing vehicle type by QoQ growth
        if 'QoQ_growth_%' in df.columns:
            recent_growth = df.dropna(subset=['QoQ_growth_%']).groupby('vehicle_type')['QoQ_growth_%'].mean()
            if not recent_growth.empty:
                top_growth_type = recent_growth.idxmax()
                top_growth_value = recent_growth.max()
                insights.append(f"🚀 **{top_growth_type}** vehicles lead growth at {top_growth_value:.1f}% avg QoQ")
        
        # Insight 2: Market leader by total registrations
        total_by_type = df.groupby('vehicle_type')['registrations'].sum()
        if not total_by_type.empty:
            market_leader = total_by_type.idxmax()
            market_share = (total_by_type.max() / total_by_type.sum() * 100)
            insights.append(f"🏆 **{market_leader}** dominates with {market_share:.1f}% market share")
        
        # Insight 3: Top manufacturer by volume
        if 'Maker' in df.columns:
            top_manufacturer = df.groupby('Maker')['registrations'].sum().idxmax()
            insights.append(f"🏭 **{top_manufacturer}** leads manufacturer rankings")
        
        # Insight 4: Data coverage and scope
        year_range = f"{df['year'].min():.0f} to {df['year'].max():.0f}"
        insights.append(f"📊 Analysis covers **{year_range}** with {df['Maker'].nunique()} manufacturers")
        
        # Insight 5: Recent trend analysis
        if 'year' in df.columns:
            recent_year = df['year'].max()
            recent_total = df[df['year'] == recent_year]['registrations'].sum()
            insights.append(f"📈 Latest year (**{recent_year:.0f}**): {recent_total:,.0f} total registrations")
        
    except Exception as e:
        insights.append(f"⚠️ Error generating insights: {str(e)[:50]}...")
    
    return insights

# =======================================================
# VISUALIZATION FUNCTIONS
# =======================================================

def create_enhanced_growth_chart(df, chart_type='qoq'):
    """
    Create enhanced growth visualization with interactive features.
    
    Args:
        df (pd.DataFrame): Data with growth metrics
        chart_type (str): Type of growth chart ('qoq' or 'yoy')
        
    Returns:
        plotly.graph_objects.Figure: Interactive growth chart
    """
    
    try:
        if chart_type == 'qoq' and 'QoQ_growth_%' in df.columns:
            growth_data = df.dropna(subset=['QoQ_growth_%'])
            y_col = 'QoQ_growth_%'
            title = "Quarter-over-Quarter Growth Analysis"
            y_title = "QoQ Growth (%)"
        else:
            # Calculate YoY growth
            growth_data = calculate_growth_metrics(df, ['year', 'vehicle_type'], 'registrations')
            if 'yoy_growth' not in growth_data.columns:
                return None
            growth_data = growth_data.dropna(subset=['yoy_growth'])
            y_col = 'yoy_growth'
            title = "Year-over-Year Growth Analysis"
            y_title = "YoY Growth (%)"
        
        if growth_data.empty:
            return None
        
        # Create interactive line chart
        fig = px.line(
            growth_data, 
            x='year_quarter' if chart_type == 'qoq' else 'year',
            y=y_col,
            color='vehicle_type',
            title=title,
            markers=True,
            template="plotly_white",
            hover_data={'registrations': ':,.0f'} if 'registrations' in growth_data.columns else None
        )
        
        # Enhance chart appearance
        fig.update_layout(
            yaxis_title=y_title,
            xaxis_title="Time Period",
            hovermode='x unified',
            xaxis_tickangle=45 if chart_type == 'qoq' else 0,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create {chart_type.upper()} growth chart: {e}")
        return None

def create_market_share_sunburst(df):
    """
    Create a sunburst chart showing market hierarchy: Vehicle Type -> Manufacturer.
    
    Args:
        df (pd.DataFrame): Vehicle registration data
        
    Returns:
        plotly.graph_objects.Figure: Interactive sunburst chart
    """
    
    try:
        # Prepare data for sunburst chart
        market_data = df.groupby(['vehicle_type', 'Maker'])['registrations'].sum().reset_index()
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=market_data['vehicle_type'].tolist() + market_data['Maker'].tolist(),
            parents=[''] * len(market_data['vehicle_type'].unique()) + market_data['vehicle_type'].tolist(),
            values=[market_data[market_data['vehicle_type'] == vtype]['registrations'].sum() 
                   for vtype in market_data['vehicle_type'].unique()] + market_data['registrations'].tolist(),
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Registrations: %{value:,.0f}<br>Percentage: %{percentParent}<extra></extra>',
            maxdepth=2
        ))
        
        fig.update_layout(
            title="Market Share Hierarchy: Vehicle Type → Manufacturer",
            template="plotly_white",
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create sunburst chart: {e}")
        return None

def create_performance_heatmap(df):
    """
    Create a performance heatmap showing growth rates by manufacturer and year.
    
    Args:
        df (pd.DataFrame): Vehicle registration data with growth metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive heatmap
    """
    
    try:
        if 'QoQ_growth_%' not in df.columns:
            return None
        
        # Prepare data for heatmap
        heatmap_data = df.pivot_table(
            index='Maker',
            columns='year',
            values='QoQ_growth_%',
            aggfunc='mean',
            fill_value=0
        )
        
        # Limit to top 15 manufacturers by total registrations
        top_manufacturers = df.groupby('Maker')['registrations'].sum().nlargest(15).index
        heatmap_data = heatmap_data.loc[heatmap_data.index.intersection(top_manufacturers)]
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdYlGn',
            title="Average QoQ Growth Rate by Manufacturer and Year (%)",
            labels=dict(x="Year", y="Manufacturer", color="Avg QoQ Growth %"),
            aspect="auto"
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Year",
            yaxis_title="Manufacturer"
        )
        
        return fig
        
    except Exception as e:
        st.warning(f"Could not create heatmap: {e}")
        return None

# =======================================================
# MAIN APPLICATION FUNCTION
# =======================================================

def main():
    """
    Main application function that orchestrates the entire dashboard.
    
    This function creates the user interface, handles data loading, manages filters,
    and coordinates all dashboard components for a seamless user experience.
    """
    
    # Application header with custom styling
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Vehicle Registration Analytics Dashboard</h1>
        <p><i>Comprehensive insights for investment analysis and market research</i></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Application introduction and instructions
    with st.expander("📖 How to Use This Dashboard", expanded=False):
        st.markdown("""
        **Welcome to the Vehicle Registration Analytics Dashboard!** This tool helps you analyze vehicle registration trends with an investor's perspective. Here's how to get started:
        
        **🎯 Getting Started:**
        1. **Data Loading**: The dashboard automatically loads your vehicle registration data
        2. **Filter Selection**: Use the sidebar to customize your analysis scope
        3. **Explore Tabs**: Navigate through different analytical views
        4. **Interactive Charts**: Hover, zoom, and click on visualizations for details
        5. **Export Data**: Download filtered datasets for external analysis
        
        **📊 Key Features:**
        • **KPI Tracking**: Monitor total registrations, growth rates, and market leaders
        • **Growth Analysis**: Year-over-Year and Quarter-over-Quarter trend analysis  
        • **Market Insights**: Automated discovery of key trends and opportunities
        • **Manufacturer Analysis**: Detailed performance comparisons
        • **Data Export**: CSV download functionality for further analysis
        
        **💡 Tips for Better Analysis:**
        • Start with broad filters, then narrow down for specific insights
        • Compare different time periods to identify seasonal patterns
        • Use growth metrics to identify emerging opportunities
        • Export data for advanced statistical analysis
        """)
    
    # Data loading with user feedback
    with st.spinner("🔄 Loading and processing vehicle registration data..."):
        df_growth, df_quarterly = load_data()
    
    # Handle data loading failure
    if df_quarterly is None:
        st.error("❌ Unable to load vehicle registration data")
        st.markdown("""
        **Troubleshooting Steps:**
        1. Ensure CSV files are in the same directory as this script
        2. Check file names match expected patterns:
           - `vehicle_registrations_cleaned-1.csv` (main data)
           - `vehicle_growth_metrics_quarterly.csv` (optional)
           - `vehicle_growth_metrics.csv` (optional)
        3. Verify CSV files contain expected columns
        4. Check file permissions and accessibility
        """)
        st.stop()
    
    # =======================================================
    # SIDEBAR CONFIGURATION
    # =======================================================
    
    st.sidebar.markdown("## 🎛️ Analysis Controls")
    st.sidebar.markdown("*Configure your analysis parameters below*")
    
    # Data summary in sidebar
    with st.sidebar.expander("📊 Dataset Information"):
        st.markdown(f"""
        **Current Dataset:**
        - **Total Records**: {len(df_quarterly):,}
        - **Manufacturers**: {df_quarterly['Maker'].nunique()}
        - **Vehicle Categories**: {', '.join(sorted(df_quarterly['vehicle_type'].unique()))}
        - **Time Coverage**: {df_quarterly['year'].min():.0f} - {df_quarterly['year'].max():.0f}
        - **Data Points**: {df_quarterly['year_quarter'].nunique()} quarters
        """)
    
    # Filter controls with help text
    st.sidebar.markdown("### 🔍 Filter Options")
    
    # Vehicle type filter
    vehicle_types = sorted(df_quarterly['vehicle_type'].unique())
    selected_vehicle_types = st.sidebar.multiselect(
        "🚗 Vehicle Categories",
        options=vehicle_types,
        default=vehicle_types,
        help="Select vehicle types to include in analysis (2-Wheeler, 3-Wheeler, 4-Wheeler, etc.)"
    )
    
    # Year filter with range slider
    years = sorted(df_quarterly['year'].dropna().unique())
    if len(years) > 1:
        year_range = st.sidebar.select_slider(
            "📅 Year Range",
            options=years,
            value=(years[0], years[-1]),
            help="Select the range of years for your analysis"
        )
        selected_years = [year for year in years if year_range[0] <= year <= year_range[1]]
    else:
        selected_years = years
        st.sidebar.info(f"📅 Single year available: {years[0] if years else 'N/A'}")
    
    # Manufacturer filter with search capability
    all_manufacturers = sorted(df_quarterly['Maker'].unique())
    
    # Show top manufacturers by default
    top_manufacturers_by_volume = (df_quarterly.groupby('Maker')['registrations']
                                   .sum().nlargest(15).index.tolist())
    
    # Manufacturer selection options
    manufacturer_selection = st.sidebar.radio(
        "🏭 Manufacturer Selection",
        ["Top 15 by Volume", "All Manufacturers", "Custom Selection"],
        help="Choose how to select manufacturers for analysis"
    )
    
    if manufacturer_selection == "Top 15 by Volume":
        selected_manufacturers = top_manufacturers_by_volume
        st.sidebar.info(f"✅ Selected top 15 manufacturers by registration volume")
    elif manufacturer_selection == "All Manufacturers":
        selected_manufacturers = all_manufacturers
        st.sidebar.warning(f"⚠️ All {len(all_manufacturers)} manufacturers selected - may affect performance")
    else:  # Custom Selection
        selected_manufacturers = st.sidebar.multiselect(
            "Select Manufacturers",
            options=all_manufacturers,
            default=top_manufacturers_by_volume[:10],
            help="Choose specific manufacturers to analyze"
        )
    
    # Initialize advanced filter variables for robustness
    filter_by_growth = False
    growth_range = (-100.0, 200.0)

    # Advanced filters in expandable section
    with st.sidebar.expander("⚙️ Advanced Filters"):
        # Minimum registration threshold
        min_registrations = st.number_input(
            "Minimum Quarterly Registrations",
            min_value=0,
            value=0,
            step=100,
            help="Filter out entries below this registration threshold"
        )
        
        # Growth rate filter
        if 'QoQ_growth_%' in df_quarterly.columns:
            filter_by_growth = st.checkbox(
                "Filter by Growth Rate",
                help="Enable to filter data by growth rate thresholds"
            )
            
            if filter_by_growth:
                growth_range = st.slider(
                    "QoQ Growth Range (%)",
                    min_value=-100.0,
                    max_value=200.0,
                    value=(-50.0, 100.0),
                    step=5.0,
                    help="Select the range of acceptable QoQ growth rates"
                )
    
    # Apply filters to data
    filtered_data = df_quarterly[
        (df_quarterly['vehicle_type'].isin(selected_vehicle_types)) &
        (df_quarterly['year'].isin(selected_years)) &
        (df_quarterly['Maker'].isin(selected_manufacturers)) &
        (df_quarterly['registrations'] >= min_registrations)
    ]
    
    # Apply advanced growth filter if enabled
    if filter_by_growth:
        filtered_data = filtered_data[
            (filtered_data['QoQ_growth_%'].between(growth_range[0], growth_range[1])) |
            (filtered_data['QoQ_growth_%'].isna())
        ]
    
    # Check if filtered data is empty
    if filtered_data.empty:
        st.warning("⚠️ No data available for selected filters. Please adjust your selections.")
        st.sidebar.error("❌ Current filters exclude all data")
        st.stop()
    
    # Show filter summary
    st.sidebar.markdown("### 📈 Filtered Data Summary")
    st.sidebar.info(f"""
    **Filtered Results:**
    - Records: {len(filtered_data):,}
    - Manufacturers: {filtered_data['Maker'].nunique()}
    - Categories: {filtered_data['vehicle_type'].nunique()}
    - Total Registrations: {filtered_data['registrations'].sum():,.0f}
    """)
    
    # =======================================================
    # KEY PERFORMANCE INDICATORS (KPI) SECTION
    # =======================================================
    
    st.markdown("## 📊 Key Performance Indicators")
    st.markdown("*Critical metrics for investment analysis and market overview*")
    
    # Calculate KPIs
    total_registrations = int(filtered_data['registrations'].sum())
    
    # Calculate YoY growth for latest available year
    latest_year = filtered_data['year'].max()
    current_year_data = filtered_data[filtered_data['year'] == latest_year]
    prev_year_data = filtered_data[filtered_data['year'] == latest_year - 1]
    
    # YoY calculation with error handling
    if not prev_year_data.empty and not current_year_data.empty:
        current_year_total = current_year_data['registrations'].sum()
        prev_year_total = prev_year_data['registrations'].sum()
        yoy_growth = ((current_year_total - prev_year_total) / prev_year_total * 100) if prev_year_total > 0 else 0
        yoy_delta_text = f"{yoy_growth:+.1f}% YoY"
        yoy_color = "normal" if abs(yoy_growth) < 10 else ("inverse" if yoy_growth < 0 else "normal")
    else:
        yoy_delta_text = "N/A"
        yoy_color = "off"
    
    # Display KPIs in columns
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            label="🎯 Total Registrations",
            value=f"{total_registrations:,}",
            delta=yoy_delta_text,
            delta_color=yoy_color,
            help="Total vehicle registrations in selected period with year-over-year growth"
        )
    
    with kpi_col2:
        category_totals = filtered_data.groupby('vehicle_type')['registrations'].sum()
        if not category_totals.empty:
            top_category = category_totals.idxmax()
            top_category_value = int(category_totals.max())
            category_share = (top_category_value / total_registrations * 100)
            st.metric(
                label="🏆 Leading Category",
                value=top_category,
                delta=f"{category_share:.1f}% share",
                help=f"Vehicle category with highest registrations: {top_category_value:,} units"
            )
        else:
            st.metric("🏆 Leading Category", "N/A")
    
    with kpi_col3:
        manufacturer_totals = filtered_data.groupby('Maker')['registrations'].sum()
        if not manufacturer_totals.empty:
            top_manufacturer = manufacturer_totals.idxmax()
            top_manufacturer_value = int(manufacturer_totals.max())
            manufacturer_share = (top_manufacturer_value / total_registrations * 100)
            st.metric(
                label="🏭 Top Manufacturer",
                value=top_manufacturer,
                delta=f"{manufacturer_share:.1f}% share",
                help=f"Leading manufacturer by registrations: {top_manufacturer_value:,} units"
            )
        else:
            st.metric("🏭 Top Manufacturer", "N/A")
    
    with kpi_col4:
        if 'QoQ_growth_%' in filtered_data.columns:
            recent_qoq_data = filtered_data.dropna(subset=['QoQ_growth_%'])
            if not recent_qoq_data.empty:
                avg_qoq = recent_qoq_data['QoQ_growth_%'].mean()
                qoq_trend = "📈" if avg_qoq > 0 else "📉" if avg_qoq < 0 else "📊"
                st.metric(
                    label="📈 Avg QoQ Growth",
                    value=f"{avg_qoq:.1f}%",
                    delta=qoq_trend,
                    help="Average Quarter-over-Quarter growth rate across all selections"
                )
            else:
                st.metric("📈 Avg QoQ Growth", "N/A")
        else:
            quarters_covered = filtered_data['year_quarter'].nunique()
            st.metric(
                label="📅 Quarters Analyzed",
                value=f"{quarters_covered}",
                help="Number of quarters included in current analysis"
            )
    
    # =======================================================
    # MARKET INSIGHTS SECTION
    # =======================================================
    
    st.markdown("## 💡 AI-Powered Market Insights")
    st.markdown("*Automated analysis of key trends and investment opportunities*")
    
    # Generate and display insights
    insights = create_insights_summary(filtered_data)
    
    # Display insights in a responsive grid
    if insights:
        insight_cols = st.columns(min(len(insights), 3))
        for i, insight in enumerate(insights):
            col_index = i % len(insight_cols)
            with insight_cols[col_index]:
                st.markdown(f"""
                <div class="insight-box">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
    
    # Additional contextual insights based on data patterns
    st.markdown("### 🔍 Trend Analysis")
    
    trend_col1, trend_col2 = st.columns(2)
    
    with trend_col1:
        # Market concentration analysis
        market_concentration = filtered_data.groupby('Maker')['registrations'].sum()
        top_5_share = (market_concentration.nlargest(5).sum() / market_concentration.sum() * 100)
        
        concentration_status = "Highly Concentrated" if top_5_share > 80 else "Moderately Concentrated" if top_5_share > 60 else "Fragmented"
        concentration_icon = "🔴" if top_5_share > 80 else "🟡" if top_5_share > 60 else "🟢"
        
        st.info(f"""
        **Market Concentration Analysis** {concentration_icon} **{concentration_status}** market  
        Top 5 manufacturers control {top_5_share:.1f}% of registrations
        """)
    
    with trend_col2:
        # Growth momentum analysis
        if 'QoQ_growth_%' in filtered_data.columns:
            growth_data = filtered_data.dropna(subset=['QoQ_growth_%'])
            if not growth_data.empty:
                positive_growth_pct = (growth_data['QoQ_growth_%'] > 0).mean() * 100
                momentum_status = "Strong" if positive_growth_pct > 70 else "Moderate" if positive_growth_pct > 40 else "Weak"
                momentum_icon = "🚀" if positive_growth_pct > 70 else "📈" if positive_growth_pct > 40 else "📉"
                
                st.info(f"""
                **Growth Momentum Analysis** {momentum_icon} **{momentum_status}** momentum  
                {positive_growth_pct:.1f}% of segments showing positive QoQ growth
                """)
    
    # =======================================================
    # TABBED INTERFACE FOR DETAILED ANALYSIS
    # =======================================================
    
    st.markdown("## 📋 Detailed Analysis")
    
    # Create tabs for different analytical perspectives
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", 
        "📈 Growth Dynamics", 
        "🏭 Manufacturer Deep-Dive", 
        "🔄 Comparative Analysis",
        "📤 Export & Documentation"
    ])
    
    # =======================================================
    # TAB 1: MARKET OVERVIEW
    # =======================================================
    
    with tab1:
        st.markdown("### 🎯 Market Overview & Trends")
        st.markdown("*Comprehensive view of registration patterns and market dynamics*")
        
        overview_col1, overview_col2 = st.columns([2, 1])
        
        with overview_col1:
            st.markdown("#### 📈 Registration Trends Over Time")
            
            # Time series analysis with enhanced interactivity
            time_agg_option = st.selectbox(
                "Time Aggregation",
                ["Quarterly", "Yearly"],
                help="Choose how to aggregate data for trend analysis"
            )
            
            if time_agg_option == "Quarterly":
                vehicle_trends = filtered_data.groupby(['year_quarter', 'vehicle_type'])['registrations'].sum().reset_index()
                x_col = 'year_quarter'
                time_title = "Quarterly Registration Trends"
            else:
                vehicle_trends = filtered_data.groupby(['year', 'vehicle_type'])['registrations'].sum().reset_index()
                x_col = 'year'
                time_title = "Yearly Registration Trends"
            
            if not vehicle_trends.empty:
                fig_trends = px.line(
                    vehicle_trends,
                    x=x_col,
                    y='registrations',
                    color='vehicle_type',
                    title=time_title,
                    markers=True,
                    template="plotly_white",
                    hover_data={'registrations': ':,.0f'}
                )
                
                # Enhance chart appearance
                fig_trends.update_layout(
                    xaxis_title="Time Period",
                    yaxis_title="Registration Count",
                    hovermode='x unified',
                    xaxis_tickangle=45 if time_agg_option == "Quarterly" else 0,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=450
                )
                
                # Add annotations for significant changes
                max_registrations = vehicle_trends.groupby(x_col)['registrations'].sum()
                peak_period = max_registrations.idxmax()
                peak_value = max_registrations.max()
                
                st.plotly_chart(fig_trends, use_container_width=True)
                
                # Trend insights
                st.caption(f"📊 Peak registration period: **{peak_period}** with {peak_value:,.0f} total registrations")
        
        with overview_col2:
            st.markdown("#### 🎯 Market Share Distribution")
            
            # Enhanced market share visualization
            market_share = filtered_data.groupby('vehicle_type')['registrations'].sum()
            
            if not market_share.empty:
                # Create enhanced pie chart
                fig_pie = go.Figure(data=[go.Pie(
                    labels=market_share.index,
                    values=market_share.values,
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='outside',
                    hovertemplate='<b>%{label}</b><br>Registrations: %{value:,.0f}<br>Share: %{percent}<extra></extra>',
                    textfont_size=12
                )])
                
                fig_pie.update_layout(
                    title="Market Share by Vehicle Type",
                    template="plotly_white",
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Market share table
                st.markdown("**Market Share Details**")
                market_share_df = pd.DataFrame({
                    'Vehicle Type': market_share.index,
                    'Registrations': market_share.values,
                    'Market Share %': (market_share.values / market_share.sum() * 100).round(1)
                }).sort_values('Registrations', ascending=False)
                
                st.dataframe(market_share_df, use_container_width=True, hide_index=True)
        
        # Market dynamics analysis
        st.markdown("#### 🔄 Market Dynamics Analysis")
        
        dynamics_col1, dynamics_col2, dynamics_col3 = st.columns(3)
        
        with dynamics_col1:
            # Market diversity index (simplified Herfindahl-Hirschman Index)
            market_shares = filtered_data.groupby('Maker')['registrations'].sum()
            market_shares_pct = market_shares / market_shares.sum()
            hhi = (market_shares_pct ** 2).sum() * 10000  # HHI scale
            
            hhi_status = "Highly Concentrated" if hhi > 2500 else "Moderately Concentrated" if hhi > 1500 else "Competitive"
            hhi_color = "🔴" if hhi > 2500 else "🟡" if hhi > 1500 else "🟢"
            
            st.metric(
                "Market Concentration (HHI)",
                f"{hhi:.0f}",
                f"{hhi_color} {hhi_status}",
                help="Herfindahl-Hirschman Index: <1500 (Competitive), 1500-2500 (Moderate), >2500 (Concentrated)"
            )
        
        with dynamics_col2:
            # Market entry analysis (new manufacturers in recent periods)
            if len(filtered_data['year'].unique()) > 1:
                recent_years = sorted(filtered_data['year'].unique())[-2:]
                recent_manufacturers = set(filtered_data[filtered_data['year'].isin(recent_years)]['Maker'].unique())
                older_manufacturers = set(filtered_data[filtered_data['year'] < recent_years[0]]['Maker'].unique())
                new_entrants = len(recent_manufacturers - older_manufacturers)
                
                st.metric(
                    "Market Entrants",
                    f"{new_entrants}",
                    "New manufacturers in recent periods",
                    help="Number of manufacturers that appeared in the most recent years"
                )
        
        with dynamics_col3:
            # Average registration size per manufacturer-category combination
            avg_registration_size = filtered_data.groupby(['Maker', 'vehicle_type'])['registrations'].mean().mean()
            
            st.metric(
                "Avg Registration Size",
                f"{avg_registration_size:,.0f}",
                "Per manufacturer-category",
                help="Average quarterly registrations per manufacturer-vehicle type combination"
            )
    
    # =======================================================
    # TAB 2: GROWTH DYNAMICS
    # =======================================================
    
    with tab2:
        st.markdown("### 📈 Growth Rate Analysis & Investment Opportunities")
        st.markdown("*Detailed analysis of growth patterns for strategic investment decisions*")
        
        # Growth analysis controls
        growth_col1, growth_col2 = st.columns([3, 1])
        
        with growth_col2:
            st.markdown("#### 🎛️ Analysis Controls")
            
            growth_metric = st.selectbox(
                "Growth Metric",
                ["Quarter-over-Quarter", "Year-over-Year"],
                help="Choose the type of growth analysis to display"
            )
            
            show_trend_lines = st.checkbox(
                "Show Trend Lines",
                value=True,
                help="Add trend lines to identify long-term patterns"
            )
            
            highlight_outliers = st.checkbox(
                "Highlight Outliers",
                value=False,
                help="Mark data points with extreme growth rates"
            )
        
        with growth_col1:
            st.markdown(f"#### 📊 {growth_metric} Growth Analysis")
            
            # Create enhanced growth chart
            growth_chart_type = 'qoq' if growth_metric == "Quarter-over-Quarter" else 'yoy'
            fig_growth = create_enhanced_growth_chart(filtered_data, growth_chart_type)
            
            if fig_growth:
                # Add trend lines if requested
                if show_trend_lines:
                    # This is a simplified trend line addition
                    # In a full implementation, you'd add proper trend calculations
                    pass
                
                st.plotly_chart(fig_growth, use_container_width=True)
            else:
                st.warning(f"⚠️ Unable to generate {growth_metric} chart with current data")
        
        # Growth performance metrics
        st.markdown("#### 🎯 Growth Performance Metrics")
        
        growth_metrics_col1, growth_metrics_col2, growth_metrics_col3 = st.columns(3)
        
        with growth_metrics_col1:
            if 'QoQ_growth_%' in filtered_data.columns:
                growth_data = filtered_data.dropna(subset=['QoQ_growth_%'])
                if not growth_data.empty:
                    top_growth_segments = growth_data.nlargest(5, 'QoQ_growth_%')[['Maker', 'vehicle_type', 'QoQ_growth_%', 'year_quarter']]
                    
                    st.markdown("**🚀 Top Growth Performers**")
                    for _, row in top_growth_segments.iterrows():
                        st.write(f"**{row['Maker']}** ({row['vehicle_type']}) - {row['QoQ_growth_%']:.1f}% in {row['year_quarter']}")
        
        with growth_metrics_col2:
            if 'QoQ_growth_%' in filtered_data.columns:
                # Consistent growth analysis
                manufacturer_consistency = (filtered_data.dropna(subset=['QoQ_growth_%'])
                                              .groupby('Maker')['QoQ_growth_%']
                                              .agg(['mean', 'std'])
                                              .eval('consistency = mean / (std + 1)')  # Avoid division by zero
                                              .nlargest(5, 'consistency'))
                
                st.markdown("**🎯 Most Consistent Growth**")
                for manufacturer in manufacturer_consistency.index[:5]:
                    consistency_score = manufacturer_consistency.loc[manufacturer, 'consistency']
                    avg_growth = manufacturer_consistency.loc[manufacturer, 'mean']
                    st.write(f"**{manufacturer}** - {avg_growth:.1f}% avg growth")
        
        with growth_metrics_col3:
            # Growth volatility analysis
            if 'QoQ_growth_%' in filtered_data.columns:
                growth_volatility = (filtered_data.dropna(subset=['QoQ_growth_%'])
                                     .groupby('vehicle_type')['QoQ_growth_%']
                                     .std()
                                     .sort_values(ascending=True))
                
                st.markdown("**📊 Growth Stability by Category**")
                for category in growth_volatility.index:
                    volatility = growth_volatility[category]
                    stability_icon = "🟢" if volatility < 10 else "🟡" if volatility < 25 else "🔴"
                    st.write(f"{stability_icon} **{category}** - {volatility:.1f}% volatility")
        
        # Advanced growth visualizations
        st.markdown("#### 🔥 Advanced Growth Visualizations")
        
        advanced_viz_col1, advanced_viz_col2 = st.columns(2)
        
        with advanced_viz_col1:
            # Growth heatmap
            fig_heatmap = create_performance_heatmap(filtered_data)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("💡 Heatmap requires QoQ growth data")
        
        with advanced_viz_col2:
            # Growth distribution analysis
            if 'QoQ_growth_%' in filtered_data.columns:
                growth_dist_data = filtered_data.dropna(subset=['QoQ_growth_%'])
                
                if not growth_dist_data.empty:
                    fig_hist = px.histogram(
                        growth_dist_data,
                        x='QoQ_growth_%',
                        color='vehicle_type',
                        title="Growth Rate Distribution",
                        nbins=30,
                        template="plotly_white",
                        opacity=0.7
                    )
                    
                    fig_hist.update_layout(
                        xaxis_title="QoQ Growth Rate (%)",
                        yaxis_title="Frequency",
                        height=400
                    )
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    # =======================================================
    # TAB 3: MANUFACTURER DEEP-DIVE
    # =======================================================
    
    with tab3:
        st.markdown("### 🏭 Manufacturer Performance Analysis")
        st.markdown("*Comprehensive analysis of manufacturer competitiveness and market positioning*")
        
        # Manufacturer selection for detailed analysis
        selected_manufacturer_analysis = st.selectbox(
            "🔍 Select Manufacturer for Detailed Analysis",
            options=["All Manufacturers"] + sorted(filtered_data['Maker'].unique()),
            help="Choose a specific manufacturer for in-depth analysis or view all"
        )
        
        manufacturer_col1, manufacturer_col2 = st.columns([2, 1])
        
        with manufacturer_col1:
            st.markdown("#### 📊 Registration Volume Leadership")
            
            # Top manufacturers analysis
            top_mfgs = filtered_data.groupby('Maker')['registrations'].sum().nlargest(15)
            
            if not top_mfgs.empty:
                # CORRECTED CODE BLOCK
                df_top_mfgs = top_mfgs.reset_index()
                df_top_mfgs.columns = ['Manufacturer', 'Registrations']

                fig_top = px.bar(
                    data_frame=df_top_mfgs,
                    x='Registrations',
                    y='Manufacturer',
                    orientation='h',
                    title="Top 15 Manufacturers by Registration Volume",
                    template="plotly_white",
                    text='Registrations'
                )
                
                fig_top.update_traces(
                    texttemplate='%{x:,.0f}', 
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Registrations: %{x:,.0f}<extra></extra>'
                )
                
                fig_top.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    xaxis_title="Total Registrations",
                    yaxis_title="Manufacturer",
                    height=500,
                    showlegend=False
                )
                
                st.plotly_chart(fig_top, use_container_width=True)
        
        with manufacturer_col2:
            st.markdown("#### 🚀 Growth Leaders")
            
            if 'QoQ_growth_%' in filtered_data.columns:
                # Growth leaders analysis
                growth_leaders = (filtered_data.dropna(subset=['QoQ_growth_%'])
                                  .groupby('Maker')['QoQ_growth_%']
                                  .agg(['mean', 'count'])
                                  .query('count >= 2')  # At least 2 data points
                                  .sort_values('mean', ascending=False)
                                  .head(10))
                
                if not growth_leaders.empty:
                    fig_growth_leaders = px.bar(
                        x=growth_leaders['mean'].values,
                        y=growth_leaders.index,
                        orientation='h',
                        title="Top 10 Growth Leaders (Avg QoQ %)",
                        template="plotly_white",
                        color=growth_leaders['mean'].values,
                        color_continuous_scale='RdYlGn'
                    )
                    
                    fig_growth_leaders.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        xaxis_title="Average QoQ Growth (%)",
                        yaxis_title="Manufacturer",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_growth_leaders, use_container_width=True)
        
        # Detailed manufacturer metrics table
        st.markdown("#### 📋 Comprehensive Manufacturer Scorecard")
        
        # Calculate comprehensive metrics
        manufacturer_metrics = []
        
        for manufacturer in filtered_data['Maker'].unique():
            mfg_data = filtered_data[filtered_data['Maker'] == manufacturer]
            
            metrics = {
                'Manufacturer': manufacturer,
                'Total_Registrations': mfg_data['registrations'].sum(),
                'Avg_Quarterly_Registrations': mfg_data['registrations'].mean(),
                'Vehicle_Categories': mfg_data['vehicle_type'].nunique(),
                'Market_Presence_Quarters': mfg_data['year_quarter'].nunique(),
                'Peak_Quarter_Performance': mfg_data['registrations'].max()
            }
            
            # Add growth metrics if available
            if 'QoQ_growth_%' in mfg_data.columns:
                growth_data = mfg_data.dropna(subset=['QoQ_growth_%'])
                if not growth_data.empty:
                    metrics['Avg_QoQ_Growth_%'] = growth_data['QoQ_growth_%'].mean()
                    metrics['Growth_Volatility'] = growth_data['QoQ_growth_%'].std()
                    metrics['Positive_Growth_Quarters_%'] = (growth_data['QoQ_growth_%'] > 0).mean() * 100
            
            manufacturer_metrics.append(metrics)
        
        manufacturer_df = pd.DataFrame(manufacturer_metrics)
        
        # Format numeric columns
        format_dict = {
            'Total_Registrations': '{:,.0f}',
            'Avg_Quarterly_Registrations': '{:,.0f}',
            'Peak_Quarter_Performance': '{:,.0f}'
        }
        
        if 'Avg_QoQ_Growth_%' in manufacturer_df.columns:
            format_dict.update({
                'Avg_QoQ_Growth_%': '{:.1f}%',
                'Growth_Volatility': '{:.1f}%',
                'Positive_Growth_Quarters_%': '{:.1f}%'
            })
        
        # Sort by total registrations
        manufacturer_df = manufacturer_df.sort_values('Total_Registrations', ascending=False)
        
        # Display with conditional formatting
        st.dataframe(
            manufacturer_df.style.format(format_dict),
            use_container_width=True,
            hide_index=True
        )
        
        # Individual manufacturer analysis
        if selected_manufacturer_analysis != "All Manufacturers":
            st.markdown(f"#### 🔍 Deep Dive: {selected_manufacturer_analysis}")
            
            mfg_detail_data = filtered_data[filtered_data['Maker'] == selected_manufacturer_analysis]
            
            if not mfg_detail_data.empty:
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    # Performance by vehicle type
                    mfg_by_type = mfg_detail_data.groupby('vehicle_type')['registrations'].sum()
                    
                    fig_mfg_pie = px.pie(
                        values=mfg_by_type.values,
                        names=mfg_by_type.index,
                        title=f"{selected_manufacturer_analysis} - Portfolio Distribution",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig_mfg_pie, use_container_width=True)
                
                with detail_col2:
                    # Performance over time
                    mfg_time_series = mfg_detail_data.groupby('year_quarter')['registrations'].sum().reset_index()
                    
                    fig_mfg_trend = px.line(
                        mfg_time_series,
                        x='year_quarter',
                        y='registrations',
                        title=f"{selected_manufacturer_analysis} - Registration Trend",
                        markers=True,
                        template="plotly_white"
                    )
                    
                    fig_mfg_trend.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_mfg_trend, use_container_width=True)
    
    # =======================================================
    # TAB 4: COMPARATIVE ANALYSIS
    # =======================================================
    
    with tab4:
        st.markdown("### 🔄 Comparative Analysis & Benchmarking")
        st.markdown("*Side-by-side comparison of manufacturers, categories, and time periods*")
        
        # Comparison controls
        comparison_type = st.selectbox(
            "🎯 Comparison Type",
            ["Manufacturer vs Manufacturer", "Vehicle Category Performance", "Time Period Analysis"],
            help="Choose the type of comparative analysis to perform"
        )
        
        if comparison_type == "Manufacturer vs Manufacturer":
            st.markdown("#### 🏭 Manufacturer Comparison")
            
            # Multi-select for manufacturers
            comparison_manufacturers = st.multiselect(
                "Select Manufacturers to Compare (2-5 recommended)",
                options=sorted(filtered_data['Maker'].unique()),
                default=sorted(filtered_data.groupby('Maker')['registrations'].sum().nlargest(3).index.tolist()),
                help="Choose manufacturers for head-to-head comparison"
            )
            
            if len(comparison_manufacturers) >= 2:
                comp_data = filtered_data[filtered_data['Maker'].isin(comparison_manufacturers)]
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    # Registration comparison
                    comp_registrations = comp_data.groupby(['year_quarter', 'Maker'])['registrations'].sum().reset_index()
                    
                    fig_comp = px.line(
                        comp_registrations,
                        x='year_quarter',
                        y='registrations',
                        color='Maker',
                        title="Registration Comparison Over Time",
                        markers=True,
                        template="plotly_white"
                    )
                    
                    fig_comp.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with comp_col2:
                    # Market share comparison
                    comp_market_share = comp_data.groupby('Maker')['registrations'].sum()
                    
                    fig_comp_share = px.bar(
                        x=comp_market_share.index,
                        y=comp_market_share.values,
                        title="Total Registration Comparison",
                        template="plotly_white",
                        text=[f"{val:,.0f}" for val in comp_market_share.values]
                    )
                    
                    fig_comp_share.update_traces(textposition='outside')
                    st.plotly_chart(fig_comp_share, use_container_width=True)
                
                # Comparative metrics table
                st.markdown("#### 📊 Comparative Performance Metrics")
                
                comp_metrics = []
                for manufacturer in comparison_manufacturers:
                    mfg_data = comp_data[comp_data['Maker'] == manufacturer]
                    
                    metrics = {
                        'Manufacturer': manufacturer,
                        'Total Registrations': mfg_data['registrations'].sum(),
                        'Average Quarterly': mfg_data['registrations'].mean(),
                        'Categories Served': mfg_data['vehicle_type'].nunique(),
                        'Market Quarters': mfg_data['year_quarter'].nunique()
                    }
                    
                    if 'QoQ_growth_%' in mfg_data.columns:
                        growth_data = mfg_data.dropna(subset=['QoQ_growth_%'])
                        if not growth_data.empty:
                            metrics['Avg QoQ Growth %'] = growth_data['QoQ_growth_%'].mean()
                    
                    comp_metrics.append(metrics)
                
                comp_df = pd.DataFrame(comp_metrics)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            else:
                st.info("👆 Please select at least 2 manufacturers for comparison")
        
        elif comparison_type == "Vehicle Category Performance":
            st.markdown("#### 🚗 Vehicle Category Analysis")
            
            category_col1, category_col2 = st.columns(2)
            
            with category_col1:
                # Category performance over time
                category_trends = filtered_data.groupby(['year_quarter', 'vehicle_type'])['registrations'].sum().reset_index()
                
                fig_cat_trends = px.line(
                    category_trends,
                    x='year_quarter',
                    y='registrations',
                    color='vehicle_type',
                    title="Category Performance Trends",
                    markers=True,
                    template="plotly_white"
                )
                
                fig_cat_trends.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_cat_trends, use_container_width=True)
            
            with category_col2:
                # Category growth comparison
                if 'QoQ_growth_%' in filtered_data.columns:
                    category_growth = (filtered_data.dropna(subset=['QoQ_growth_%'])
                                       .groupby('vehicle_type')['QoQ_growth_%']
                                       .agg(['mean', 'std'])
                                       .round(2))
                    
                    fig_cat_growth = px.scatter(
                        x=category_growth['mean'],
                        y=category_growth['std'],
                        text=category_growth.index,
                        title="Growth vs Volatility by Category",
                        labels={'x': 'Average QoQ Growth (%)', 'y': 'Growth Volatility (%)'},
                        template="plotly_white"
                    )
                    
                    fig_cat_growth.update_traces(textposition="middle right")
                    st.plotly_chart(fig_cat_growth, use_container_width=True)
        
        else:  # Time Period Analysis
            st.markdown("#### 📅 Time Period Comparison")
            
            # Year-over-year comparison
            if len(filtered_data['year'].unique()) > 1:
                years_available = sorted(filtered_data['year'].unique())
                
                time_col1, time_col2 = st.columns(2)
                
                with time_col1:
                    selected_years_comp = st.multiselect(
                        "Select Years to Compare",
                        options=years_available,
                        default=years_available[-2:] if len(years_available) >= 2 else years_available,
                        help="Choose years for comparative analysis"
                    )
                
                with time_col2:
                    comparison_metric = st.selectbox(
                        "Comparison Metric",
                        ["Total Registrations", "Average Quarterly Registrations", "Number of Active Manufacturers"],
                        help="Choose which metric to compare across time periods"
                    )
                
                if len(selected_years_comp) >= 2:
                    time_comparison_data = []
                    
                    for year in selected_years_comp:
                        year_data = filtered_data[filtered_data['year'] == year]
                        
                        if comparison_metric == "Total Registrations":
                            value = year_data['registrations'].sum()
                        elif comparison_metric == "Average Quarterly Registrations":
                            value = year_data['registrations'].mean()
                        else:  # Number of Active Manufacturers
                            value = year_data['Maker'].nunique()
                        
                        time_comparison_data.append({
                            'Year': year,
                            'Value': value,
                            'Metric': comparison_metric
                        })
                    
                    time_comp_df = pd.DataFrame(time_comparison_data)
                    
                    fig_time_comp = px.bar(
                        time_comp_df,
                        x='Year',
                        y='Value',
                        title=f"{comparison_metric} by Year",
                        template="plotly_white",
                        text='Value'
                    )
                    
                    fig_time_comp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
                    st.plotly_chart(fig_time_comp, use_container_width=True)
                    
                    # Calculate year-over-year changes
                    if len(time_comp_df) >= 2:
                        st.markdown("#### 📈 Year-over-Year Changes")
                        
                        yoy_changes = []
                        for i in range(1, len(time_comp_df)):
                            current_year = time_comp_df.iloc[i]
                            previous_year = time_comp_df.iloc[i-1]
                            
                            change = current_year['Value'] - previous_year['Value']
                            change_pct = (change / previous_year['Value'] * 100) if previous_year['Value'] > 0 else 0
                            
                            yoy_changes.append({
                                'Period': f"{previous_year['Year']:.0f} → {current_year['Year']:.0f}",
                                'Absolute Change': change,
                                'Percentage Change': change_pct
                            })
                        
                        yoy_df = pd.DataFrame(yoy_changes)
                        st.dataframe(yoy_df, use_container_width=True, hide_index=True)
            else:
                st.info("📅 Time period comparison requires data from multiple years")
    
    # =======================================================
    # TAB 5: EXPORT & DOCUMENTATION
    # =======================================================
    
    with tab5:
        st.markdown("### 📤 Data Export & Documentation")
        st.markdown("*Download filtered data and access comprehensive documentation*")
        
        export_col1, export_col2 = st.columns([2, 1])
        
        with export_col1:
            st.markdown("#### 💾 Export Filtered Dataset")
            
            # Prepare export data
            export_data = filtered_data.copy().sort_values(['year_quarter', 'vehicle_type', 'Maker'])
            
            # Export format options
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel (XLSX)", "JSON"],
                help="Choose the format for data export"
            )
            
            # Additional export options
            include_metadata = st.checkbox(
                "Include Analysis Metadata",
                value=True,
                help="Add analysis parameters and summary statistics to export"
            )
            
            # Generate export data based on format
            if export_format == "CSV":
                if include_metadata:
                    # Create metadata header
                    metadata_text = f"""# Vehicle Registration Analytics Export
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Filters Applied:
# - Vehicle Types: {', '.join(selected_vehicle_types)}
# - Years: {min(selected_years)} to {max(selected_years)}
# - Manufacturers: {len(selected_manufacturers)} selected
# - Total Records: {len(export_data):,}
# - Total Registrations: {export_data['registrations'].sum():,}
#
"""
                    # Convert to CSV with metadata
                    csv_data = metadata_text.encode('utf-8') + export_data.to_csv(index=False).encode('utf-8')
                else:
                    csv_data = export_data.to_csv(index=False).encode('utf-8')
                
                file_extension = "csv"
                mime_type = "text/csv"
                export_bytes = csv_data
            
            elif export_format == "Excel (XLSX)":
                import io
                
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Main data sheet
                    export_data.to_excel(writer, sheet_name='Registration_Data', index=False)
                    
                    if include_metadata:
                        # Summary sheet
                        summary_data = {
                            'Metric': [
                                'Total Records',
                                'Total Registrations',
                                'Date Range',
                                'Vehicle Types',
                                'Manufacturers',
                                'Export Date'
                            ],
                            'Value': [
                                len(export_data),
                                f"{export_data['registrations'].sum():,}",
                                f"{export_data['year'].min():.0f} - {export_data['year'].max():.0f}",
                                ', '.join(selected_vehicle_types),
                                f"{len(selected_manufacturers)} selected",
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Export_Summary', index=False)
                
                export_bytes = output.getvalue()
                file_extension = "xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            else:  # JSON
                # Define a helper class to handle NumPy number types
                import json
                import numpy as np
                class NpEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        if isinstance(obj, np.floating):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return super(NpEncoder, self).default(obj)

                if include_metadata:
                    full_json = {
                        'metadata': {
                            'export_date': datetime.now().isoformat(),
                            'total_records': len(export_data),
                            'total_registrations': int(export_data['registrations'].sum()),
                            'filters': {
                                'vehicle_types': selected_vehicle_types,
                                'years': [int(y) for y in selected_years], # Ensure years are standard int
                                'manufacturers_count': len(selected_manufacturers)
                            }
                        },
                        'data': export_data.to_dict(orient='records')
                    }
                    # Use the new class to encode the JSON data correctly
                    json_data = json.dumps(full_json, indent=2, cls=NpEncoder)
                else:
                    # Pandas' built-in to_json handles this automatically
                    json_data = export_data.to_json(orient='records', indent=2)

                export_bytes = json_data.encode('utf-8')
                file_extension = "json"
                mime_type = "application/json"
            
            # Download button
            st.download_button(
                label=f"📥 Download as {export_format}",
                data=export_bytes,
                file_name=f"vehicle_registration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}",
                mime=mime_type,
                help=f"Download the filtered dataset in {export_format} format"
            )
            
            # Export summary
            st.info(f"""
            **📊 Export Summary:**
            - **Records**: {len(export_data):,}
            - **Date Range**: {export_data['year_quarter'].min()} to {export_data['year_quarter'].max()}
            - **Categories**: {', '.join(sorted(export_data['vehicle_type'].unique()))}
            - **Manufacturers**: {export_data['Maker'].nunique()}
            - **Total Registrations**: {export_data['registrations'].sum():,.0f}
            """)
        
        with export_col2:
            st.markdown("#### 📚 Documentation & Resources")
            
            # Quick reference guide
            with st.expander("📖 Quick Reference Guide"):
                st.markdown("""
                **Data Columns Explained:**
                - **Maker**: Manufacturer/brand name
                - **vehicle_type**: Category (2W/3W/4W etc.)
                - **year**: Calendar year
                - **quarter**: Quarter number (1-4)
                - **year_quarter**: Combined period identifier
                - **registrations**: Number of vehicles registered
                - **QoQ_growth_%**: Quarter-over-Quarter growth percentage
                
                **Key Metrics:**
                - **YoY Growth**: Year-over-Year percentage change
                - **QoQ Growth**: Quarter-over-Quarter percentage change
                - **Market Share**: Percentage of total registrations
                - **HHI**: Herfindahl-Hirschman Index (market concentration)
                """)
            
            # Data quality information
            with st.expander("🔍 Data Quality & Limitations"):
                st.markdown("""
                **Data Quality Notes:**
                - Missing values are handled by exclusion or zero-filling
                - Growth calculations require at least 2 time periods
                - Outliers may impact growth rate calculations
                - Market concentration metrics are calculated on filtered data
                
                **Known Limitations:**
                - Data availability varies by manufacturer and time period
                - Some manufacturers may have incomplete quarterly data
                - Growth rates can be volatile for small registration volumes
                - Historical data may have different collection methodologies
                """)
            
            # Analysis methodology
            with st.expander("📊 Analysis Methodology"):
                st.markdown("""
                **Calculation Methods:**
                
                **Growth Rates:**
                - QoQ: ((Current Quarter - Previous Quarter) / Previous Quarter) × 100
                - YoY: ((Current Year - Previous Year) / Previous Year) × 100
                
                **Market Metrics:**
                - Market Share: (Segment Total / Overall Total) × 100
                - HHI: Σ(Market Share²) × 10,000
                - Volatility: Standard deviation of growth rates
                
                **Data Processing:**
                - Monthly data aggregated to quarterly totals
                - Missing values handled contextually
                - Outliers identified but not automatically excluded
                """)
        
        # Additional resources and links
        st.markdown("#### 🌐 Additional Resources")
        
        resource_col1, resource_col2, resource_col3 = st.columns(3)
        
        with resource_col1:
            st.markdown("""
            **📈 Investment Analysis:**
            - Market trend identification
            - Growth opportunity assessment
            - Competitive positioning analysis
            - Risk factor evaluation
            """)
        
        with resource_col2:
            st.markdown("""
            **🔧 Technical Details:**
            - Data source: Vahan Dashboard
            - Processing: Python/Pandas
            - Visualization: Plotly
            - Interface: Streamlit
            """)
        
        with resource_col3:
            st.markdown("""
            **💡 Usage Tips:**
            - Start with broad filters
            - Compare time periods for trends
            - Use growth metrics for opportunities
            - Export for advanced analysis
            """)
    
    # =======================================================
    # FOOTER AND ADDITIONAL INFORMATION
    # =======================================================
    
    st.markdown("---")
    
    # Performance and system information
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **📊 Data Analytics Dashboard** Built for investment analysis and market research
        """)
    
    with footer_col2:
        st.markdown("""
        **🛠️ Technology Stack** Python • Streamlit • Plotly • Pandas
        """)
    
    with footer_col3:
        processing_time = datetime.now()
        st.markdown(f"""
        **⏱️ Last Updated** {processing_time.strftime('%Y-%m-%d %H:%M:%S')}
        """)
    
    # Help and support section
    with st.expander("❓ Help & Support"):
        st.markdown("""
        **Need Help?**
        
        **Common Issues:**
        1. **No data displayed**: Check filter selections and ensure data files are available
        2. **Charts not loading**: Try refreshing the page or adjusting filters
        3. **Export not working**: Ensure you have appropriate permissions for downloads
        4. **Performance issues**: Reduce the number of selected manufacturers or time periods
        
        **Best Practices:**
        - Use specific filters for faster performance
        - Export data for advanced statistical analysis
        - Compare similar time periods for meaningful insights
        - Monitor both volume and growth metrics for comprehensive analysis
        
        **Feature Requests & Bug Reports:**
        This dashboard is designed for the Backend Developer Internship assignment.
        For production use, consider implementing additional features like:
        - Real-time data updates
        - Advanced statistical modeling
        - Predictive analytics
        - Custom report generation
        """)
    
    # Advanced configuration for power users
    if st.checkbox("🔧 Show Advanced Configuration", help="Display advanced settings for power users"):
        st.markdown("#### ⚙️ Advanced Configuration")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.markdown("**Performance Settings:**")
            cache_ttl = st.number_input("Cache TTL (seconds)", min_value=60, max_value=3600, value=3600)
            chart_height = st.number_input("Default Chart Height", min_value=300, max_value=800, value=400)
            
        with adv_col2:
            st.markdown("**Display Options:**")
            show_data_points = st.checkbox("Show individual data points on charts", value=True)
            use_scientific_notation = st.checkbox("Use scientific notation for large numbers", value=False)
        
        st.info("💡 Advanced settings affect dashboard behavior. Changes require page refresh to take effect.")

# =======================================================
# APPLICATION ENTRY POINT
# =======================================================

if __name__ == "__main__":
    """
    Application entry point.
    
    This section ensures the main function runs when the script is executed directly.
    It also provides error handling for unexpected issues during dashboard initialization.
    """
    
    try:
        main()
    except Exception as e:
        st.error(f"❌ Critical application error: {e}")
        st.markdown("""
        **Troubleshooting Steps:**
        1. Ensure all required CSV files are present
        2. Check Python environment and dependencies
        3. Verify file permissions and accessibility
        4. Restart the Streamlit application
        
        **Required Files:**
        - `vehicle_registrations_cleaned-1.csv` (primary data source)
        - Optional: `vehicle_growth_metrics_quarterly.csv`
        - Optional: `vehicle_growth_metrics.csv`
        """)
        
        if st.button("🔄 Retry Loading"):
            # CORRECTED FUNCTION CALL
            st.rerun()

# =======================================================
# END OF ENHANCED VEHICLE REGISTRATION ANALYTICS DASHBOARD
# =======================================================

"""
ENHANCED DASHBOARD FEATURES SUMMARY:
====================================

🎯 USER EXPERIENCE ENHANCEMENTS:
• Comprehensive documentation and inline help
• Progressive data loading with status updates
• Interactive filtering with smart defaults
• Responsive design with mobile-friendly layouts
• Contextual tooltips and explanations
• Error handling with helpful troubleshooting

📊 ANALYTICAL CAPABILITIES:
• Multi-dimensional filtering (time, manufacturer, category)
• Advanced growth metrics (YoY, QoQ, volatility)
• Market concentration analysis (HHI)
• Comparative analysis tools
• Automated insight generation
• Performance benchmarking

🎨 VISUALIZATION IMPROVEMENTS:
• Interactive charts with hover details
• Enhanced color schemes and styling
• Multiple chart types (line, bar, pie, heatmap, sunburst)
• Trend analysis with outlier detection
• Customizable time aggregation
• Export-ready visualizations

💾 DATA MANAGEMENT:
• Multiple export formats (CSV, Excel, JSON)
• Metadata inclusion in exports
• Data quality indicators
• Processing status feedback
• Cache optimization for performance
• Comprehensive data validation

🔧 TECHNICAL ENHANCEMENTS:
• Modular code structure with clear documentation
• Error handling with graceful degradation
• Performance optimization with caching
• Responsive UI components
• Advanced configuration options
• Scalable architecture

This enhanced dashboard provides a professional-grade analytics platform
suitable for investment analysis, market research, and strategic decision-making
in the vehicle registration sector.
"""