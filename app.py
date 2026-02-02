import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analytics_functions import (
    customer_traffic_analysis,
    conversion_rate_analysis,
    sales_amount_analysis,
    sales_quantity_analysis,
    sales_person,
    truck_type
)

# Page basic configuration
st.set_page_config(page_title="Truck Sales Data Analysis System | Dashboard", page_icon="📊", layout="wide")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'none'

# Global color palette
COLOR_PRIMARY = '#2E86AB'
COLOR_SUCCESS = '#C73E1D'
COLOR_WARNING = '#F18F01'
COLOR_SECOND = '#A23B72'
COLOR_GRAY = '#f0f2f6'

# Load data with caching to improve performance
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("data.xlsx")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Total sales'] = pd.to_numeric(df['Total sales'], errors='coerce').fillna(0)
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce').fillna(0)
        df['Customer Traffic'] = pd.to_numeric(df['Customer Traffic'], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        st.error("❌ Error: [data.xlsx] file not found in the same directory!")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Calculate core KPIs (sales figures)
def calculate_kpi(df):
    if df.empty: return {}
    today = df['Date'].max()
    week_start = today - pd.Timedelta(days=7)
    month_start = today - pd.Timedelta(days=30)

    # Filter data for weekly and monthly periods
    df_week = df[(df['Date'] >= week_start) & (df['Date'] <= today)]
    df_month = df[(df['Date'] >= month_start) & (df['Date'] <= today)]

    # Sum up sales amounts
    week_sales = df_week['Total sales'].sum()
    month_sales = df_month['Total sales'].sum()

    return {
        'week_sales': week_sales, 'month_sales': month_sales,
        'today': today.strftime('%Y-%m-%d'), 'week_start': week_start.strftime('%Y-%m-%d'),
        'month_start': month_start.strftime('%Y-%m-%d')
    }

kpi_data = calculate_kpi(df)

# ==================== Sidebar Navigation ====================
st.sidebar.title("📊 Data Analysis Navigation")

# Dashboard button logic
if st.sidebar.button("🏠 Panoramic Data Dashboard", use_container_width=True, type="primary"):
    st.query_params.clear()
    st.query_params["page"] = "dashboard"
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Detailed Data Analysis")

# Define analysis module mapping
analysis_modules = {
    "📈 Customer Traffic Analysis": {
        "func": customer_traffic_analysis,
        "dimensions": ["daily", "weekly", "monthly", "week_growth", "month_growth"],
        "titles": ["Daily Traffic", "Weekly Traffic", "Monthly Traffic", "Weekly Growth Rate", "Monthly Growth Rate"]
    },
    "💹 Conversion Rate & Avg Daily Amount": {
        "func": conversion_rate_analysis,
        "dimensions": ["daily", "weekly", "monthly"],
        "titles": ["Daily Conversion Rate", "Weekly Conversion Rate & Avg Sales", "Monthly Conversion Rate & Avg Sales"]
    },
    "💰 Sales Amount Analysis": {
        "func": sales_amount_analysis,
        "dimensions": ["daily", "weekly", "monthly", "week_growth", "month_growth"],
        "titles": ["Daily Sales Amount", "Weekly Sales Amount", "Monthly Sales Amount", "Weekly Growth Rate", "Monthly Growth Rate"]
    },
    "📦 Sales Quantity Analysis": {
        "func": sales_quantity_analysis,
        "dimensions": ["daily", "weekly", "monthly", "week_growth", "month_growth"],
        "titles": ["Daily Sales Quantity", "Weekly Sales Quantity", "Monthly Sales Quantity", "Weekly Growth Rate", "Monthly Growth Rate"]
    },
    "👨💼 Sales Consultant Performance": {
        "func": sales_person,
        "dimensions": ["total", "week", "daily"],
        "titles": ["Total Sales Share", "Weekly Performance", "Daily Performance"]
    },
    "🚛 Truck Type Sales Analysis": {
        "func": truck_type,
        "dimensions": ["total"],
        "titles": ["Total Sales Quantity by Truck Type"]
    }
}

# Analysis module selection dropdown
analysis_module = st.sidebar.selectbox(
    "Select Analysis Module",
    list(analysis_modules.keys()),
    index=0,
    disabled=False
)

# Load selected analysis button
if st.sidebar.button("📝 Load Selected Analysis", use_container_width=True):
    st.query_params["page"] = "detail"
    st.query_params["analysis"] = analysis_module
    st.rerun()

# Get current page and selected analysis module
current_page = st.query_params.get("page", "dashboard")
selected_detail_analysis = st.query_params.get("analysis", analysis_module)
selected_analysis_config = analysis_modules.get(selected_detail_analysis, list(analysis_modules.values())[0])

# ==================== Main Dashboard Section ====================
if not df.empty:
    if current_page == "dashboard":
        # Initial dashboard trigger button
        if st.button("🚀 Start Panoramic Analysis", type="primary", use_container_width=True):
            st.title("📊 Truck Sales Panoramic Data Dashboard")
            st.markdown(f"**📅 Data Updated to: {kpi_data.get('today', 'No Data')}**")
            st.divider()

            # 1. Core KPI Metrics (2 columns layout)
            st.subheader("💎 Core Performance Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                # Weekly sales KPI card
                st.markdown(f"""
                    <div style='background-color:{COLOR_GRAY};padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                        <p style='color:{COLOR_PRIMARY};font-size:14px;margin:0;'>Sales in the past week ({kpi_data.get('week_start')} ~ {kpi_data.get('today')})</p>
                        <p style='font-size:28px;font-weight:bold;margin:5px 0;color:#212529;'>{kpi_data.get('week_sales', 0):,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Monthly sales KPI card
                st.markdown(f"""
                    <div style='background-color:{COLOR_GRAY};padding:20px;border-radius:10px;text-align:center;box-shadow:0 2px 4px rgba(0,0,0,0.1);'>
                        <p style='color:{COLOR_SUCCESS};font-size:14px;margin:0;'>Sales in the past month ({kpi_data.get('month_start')} ~ {kpi_data.get('today')})</p>
                        <p style='font-size:28px;font-weight:bold;margin:5px 0;color:#212529;'>{kpi_data.get('month_sales', 0):,.0f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            st.divider()

            # 2. Sales & Inventory Overview (2 columns layout)
            st.subheader("📈 Sales & Inventory Overview")
            col1, col2 = st.columns(2, gap='large')

            with col1:
                st.markdown("### 🚛 Sold Units by Model")
                truck_type(df, plot_type="total")
                st.pyplot(plt, use_container_width=True)
                plt.clf()

            with col2:
                st.markdown("### 💰 Daily Sales Trend")
                sales_amount_analysis(df, analysis_type="daily")
                st.pyplot(plt, use_container_width=True)
                plt.clf()

            st.divider()

            # 3. Growth & Traffic Analysis (2 columns layout)
            st.subheader("📊 Growth & Traffic Analysis")
            col1, col2 = st.columns(2, gap='large')

            with col1:
                st.markdown("### 📈 Weekly Sales Growth")
                sales_amount_analysis(df, analysis_type="week_growth")
                st.pyplot(plt, use_container_width=True)
                plt.clf()

            with col2:
                st.markdown("### 🧑🤝🧑 Daily Customer Traffic")
                customer_traffic_analysis(df, analysis_type="daily")
                st.pyplot(plt, use_container_width=True)
                plt.clf()

            st.divider()

            # 4. Operational Efficiency (full width for dual-axis chart)
            st.subheader("✅ Operational Efficiency")
            st.markdown("### 📊 Weekly Conversion Rate & Avg Sales")
            conversion_rate_analysis(df, analysis_type="weekly")
            st.pyplot(plt, use_container_width=True)
            plt.clf()

            st.divider()
            st.success("✅ Panoramic Data Analysis Completed!")

    # ==================== Detailed Analysis Section ====================
    else:
        st.title(f"🔍 {selected_detail_analysis}")
        st.markdown(f"**📅 Data Updated to: {kpi_data.get('today', 'No Data')}**")
        st.divider()

        # Get current module configuration
        analysis_func = selected_analysis_config["func"]
        dimensions = selected_analysis_config["dimensions"]
        titles = selected_analysis_config["titles"]

        # Display all charts for selected module (one per row)
        for i in range(0, len(dimensions)):
            st.subheader(f"📊 {titles[i]}")

            # Parameter name adaptation for different modules
            if selected_detail_analysis == "🚛 Truck Type Sales Analysis":
                analysis_func(df, plot_type=dimensions[i])
            elif selected_detail_analysis == "👨💼 Sales Consultant Performance":
                analysis_func(df, type_input=dimensions[i])
            else:
                analysis_func(df, analysis_type=dimensions[i])

            # Render plot and clear figure for next iteration
            st.pyplot(plt, use_container_width=True)
            plt.clf()
            st.divider()

        st.success(f"✅ {selected_detail_analysis} - All Charts Loaded Successfully!")
else:
    st.warning("⚠️ Please make sure the [data.xlsx] file exists and loaded successfully!")
