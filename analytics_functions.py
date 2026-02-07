import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Global Configuration 
# Fix Chinese character/negative sign display issues
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Standard business color palette
COLOR_PRIMARY = '#2E86AB'    # Deep blue
COLOR_SECOND = '#A23B72'     # Magenta
COLOR_WARNING = '#F18F01'    # Orange yellow
COLOR_SUCCESS = '#C73E1D'    # Burgundy
COLOR_GRAY = '#f0f2f6'       # Light gray

# Currency formatter with thousand separators
def format_currency(x, pos):
    return f'{x:,.0f}'

#  Sales Performance by Salesperson 
def sales_person(df, type_input="total"):
    """
    Sales analysis visualization 
    - total: Pie chart of total sales by salesperson
    - week: Bar chart of weekly sales by salesperson
    - daily: Line chart of daily sales by salesperson
    
    Key features:
    - Week axis shows date range (e.g., 01.01-01.07)
    - Optimized performance
    - All values use thousand separators
    """
    # Basic data validation to prevent errors
    df['Date'] = pd.to_datetime(df['Date'])
    df['Total sales'] = pd.to_numeric(df['Total sales'], errors='coerce').fillna(0)
    business_colors = [COLOR_PRIMARY, COLOR_SECOND, COLOR_WARNING, COLOR_SUCCESS]

    # Total sales - Pie chart
    if type_input == 'total':
        sales_total = df.groupby('Salesperson')['Total sales'].sum().reset_index()
        plt.figure(figsize=(8,6), dpi=90)
        
        # Custom percentage formatter for pie chart
        def make_autopct(values):
            def my_autopct(pct):
                val = int(round(pct*sum(values)/100.0))
                return f'{pct:.1f}%\n{val:,}'
            return my_autopct
        
        wedges, _, autotexts = plt.pie(
            sales_total['Total sales'], 
            labels=sales_total['Salesperson'], 
            autopct=make_autopct(sales_total['Total sales']),
            startangle=90, 
            explode=[0.01]*len(sales_total), 
            colors=business_colors[:len(sales_total)], 
            textprops={'fontsize':9, 'weight':'bold'}
        )
        [t.set_color('white') for t in autotexts]
        plt.title('Total Sales by Salesperson', fontsize=13, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()

    # Weekly sales - Bar chart
    elif type_input == 'week':
        df['Week'] = df['Date'].dt.to_period('W')
        week_sales = df.groupby(['Week','Salesperson'])['Total sales'].sum().reset_index()
        # Format week range to compact display (e.g., 01.01-01.07)
        week_sales['Week Range'] = week_sales['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        week_pivot = week_sales.pivot(index='Week Range', columns='Salesperson', values='Total sales').fillna(0)

        plt.figure(figsize=(10,6), dpi=90)
        ax = week_pivot.plot(
            kind='bar', 
            width=0.65, 
            color=business_colors[:len(week_pivot.columns)], 
            edgecolor='white', 
            linewidth=1,
            ax=plt.gca()
        )
        ax.set_title('Weekly Sales by Salesperson', fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Week Date Range', fontsize=11)
        ax.set_ylabel('Total Sales Amount', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.legend(title='Salesperson', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
        ax.grid(axis='y', alpha=0.2)
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))

        # Add value labels on top of bars
        for c in ax.containers:
            ax.bar_label(c, fmt='{:,.0f}', fontsize=8, weight='bold', padding=3)
        plt.tight_layout()

    # Daily sales - Line chart
    elif type_input == 'daily':
        df['Date Formatted'] = df['Date'].dt.strftime('%Y-%m-%d')
        day_sales = df.groupby(['Date Formatted','Salesperson'])['Total sales'].sum().reset_index()
        day_pivot = day_sales.pivot(index='Date Formatted', columns='Salesperson', values='Total sales').fillna(0)

        plt.figure(figsize=(10,6), dpi=90)
        ax = day_pivot.plot(
            kind='line', 
            linewidth=2.5, 
            marker='o', 
            markersize=6, 
            color=business_colors[:len(day_pivot.columns)],
            ax=plt.gca()
        )
        ax.set_title('Daily Sales by Salesperson', fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Total Sales Amount', fontsize=11)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.legend(title='Salesperson', bbox_to_anchor=(1.01,1), loc='upper left', frameon=False)
        ax.grid(axis='y', alpha=0.2)
        ax.yaxis.set_major_formatter(FuncFormatter(format_currency))

        # Add value labels on data points
        for line in ax.get_lines():
            [ax.annotate(
                f'{int(y):,}', 
                xy=(x,y), 
                xytext=(0,5), 
                textcoords='offset points', 
                fontsize=8, 
                weight='bold', 
                ha='center'
            ) for x,y in zip(line.get_xdata(), line.get_ydata())]
        plt.tight_layout()

    else:
        print(f"Input error! Only 'total'/'week'/'daily' are supported. Current input: {type_input}")

#  Sales Analysis by Truck Type 
def truck_type(df, plot_type: str = "total", start_date=None, end_date=None):
    """
    Truck type sales analysis visualization
    
    Parameters:
    - df: Source data (must contain 'Truck Type' and 'Sales' columns; 'Date' for time filtering)
    - plot_type: "total" (full dataset) or "time_range" (filtered by date)
    - start_date/end_date: Required for time_range (format: '2024-01-01')
    
    Output: Bar chart showing total sales quantity by truck type with percentage breakdown
    """
    # Business style configuration
    business_colors = [COLOR_PRIMARY, COLOR_SECOND, COLOR_WARNING, COLOR_SUCCESS, '#6A994E', '#577590']
    plt.figure(figsize=(9, 6), dpi=100)

    # Data preparation - full data or time range filter
    df_analysis = df.copy()
    if plot_type == "time_range":
        if 'Date' not in df_analysis.columns:
            print("⚠️ Data validation failed: 'Date' column missing - cannot apply time filter!")
            return
        if start_date is None or end_date is None:
            print("⚠️ Data validation failed: Please provide both start and end dates!")
            return

        df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_analysis = df_analysis[(df_analysis['Date'] >= start_dt) & (df_analysis['Date'] <= end_dt)]

    # Calculate total sales by truck type
    truck_total = df_analysis.groupby('Truck Type')['Sales'].sum().reset_index()

    # Handle empty dataset scenario
    if truck_total.empty or truck_total['Sales'].sum() == 0:
        print("ℹ️ Info: No valid sales data found for the selected criteria - cannot generate chart!")
        return

    # Add percentage calculation and sort by sales volume
    total_sales = truck_total['Sales'].sum()
    truck_total['Percentage (%)'] = round(truck_total['Sales'] / total_sales * 100, 1)
    truck_total = truck_total.sort_values('Sales', ascending=False)

    # Create bar chart
    x = truck_total['Truck Type']
    y = truck_total['Sales']
    bar_colors = business_colors[:len(truck_total)]

    bars = plt.bar(x, y, color=bar_colors, edgecolor='white', linewidth=1.2, alpha=0.95)

    # Add dual labels (formatted number + percentage) on top of bars
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        sales_num = truck_total['Sales'].iloc[idx]
        sales_pct = truck_total['Percentage (%)'].iloc[idx]
        label_text = f'{sales_num:,.0f}\n({sales_pct}%)'
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + max(y)*0.01,
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=10, 
            fontweight='bold', 
            color='#212529'
        )

    # Chart styling
    plt.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.8)
    plt.ylabel('Total Sales Quantity', fontsize=12, fontweight='bold')
    plt.xlabel('Truck Type', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10)

    # Dynamic title based on plot type
    if plot_type == "total":
        plt.title('Total Sales Quantity by Truck Type', fontsize=14, fontweight='bold', pad=20)
    else:
        plt.title(
            f'Total Sales Quantity by Truck Type\n({start_date} ~ {end_date})',
            fontsize=14, 
            fontweight='bold', 
            pad=20
        )

    plt.tight_layout()

# Customer Traffic Analysis 
def customer_traffic_analysis(df, analysis_type="daily"):
    """
    Comprehensive customer traffic analysis
    
    analysis_type options:
    - daily: Daily traffic trend
    - weekly: Weekly traffic trend (shows date range on x-axis)
    - monthly: Monthly traffic trend
    - week_growth: Weekly growth rate
    - month_growth: Monthly growth rate
    """
    df_ana = df.copy()
    df_ana['Date'] = pd.to_datetime(df_ana['Date'])
    plt.figure(figsize=(10,6), dpi=90)

    # Daily customer traffic
    if analysis_type == 'daily':
        data = df_ana.groupby('Date')['Customer Traffic'].sum().reset_index()
        plt.plot(data['Date'], data['Customer Traffic'], COLOR_PRIMARY, lw=2, marker='o', ms=5)
        # Add value labels
        for _, r in data.iterrows():
            plt.text(
                r['Date'], 
                r['Customer Traffic']+0.8, 
                f"{r['Customer Traffic']}", 
                ha='center', 
                fontsize=9, 
                weight='bold', 
                c=COLOR_PRIMARY
            )
        plt.title('Daily Customer Traffic Trend', fontsize=14, weight='bold')
        plt.ylabel('Customer Traffic', fontsize=12)
        plt.xlabel('Date', fontsize=12)

    # Weekly customer traffic (formatted date range on x-axis)
    elif analysis_type == 'weekly':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        week_data = df_ana.groupby('Week')['Customer Traffic'].sum().reset_index()
        # Format week range to compact display (e.g., 01.01-01.07)
        week_data['Week Range'] = week_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        plt.plot(week_data['Week Range'], week_data['Customer Traffic'], COLOR_SECOND, lw=2, marker='o', ms=5)
        # Add value labels
        for i, v in enumerate(week_data['Customer Traffic']):
            plt.text(
                i, 
                v+1.5, 
                f"{v}", 
                ha='center', 
                fontsize=9, 
                weight='bold', 
                c=COLOR_SECOND
            )
        plt.title('Weekly Customer Traffic Trend', fontsize=14, weight='bold')
        plt.ylabel('Customer Traffic', fontsize=12)
        plt.xlabel('Week Date Range', fontsize=12)

    # Monthly customer traffic
    elif analysis_type == 'monthly':
        data = df_ana.groupby(df_ana['Date'].dt.strftime('%Y-%m'))['Customer Traffic'].sum().reset_index()
        plt.plot(data.iloc[:,0], data.iloc[:,1], COLOR_WARNING, lw=2, marker='o', ms=5)
        # Add value labels
        for i, v in enumerate(data.iloc[:,1]):
            plt.text(
                i, 
                v+2, 
                f"{v}", 
                ha='center', 
                fontsize=9, 
                weight='bold', 
                c=COLOR_WARNING
            )
        plt.title('Monthly Customer Traffic Trend', fontsize=14, weight='bold')
        plt.ylabel('Customer Traffic', fontsize=12)
        plt.xlabel('Year-Month', fontsize=12)

    # Weekly growth rate
    elif analysis_type == 'week_growth':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        week_data = df_ana.groupby('Week')['Customer Traffic'].sum().reset_index()
        week_data['Week Range'] = week_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        week_data['Growth Rate'] = round(week_data['Customer Traffic'].pct_change()*100, 1)
        plt.plot(week_data['Week Range'], week_data['Growth Rate'], COLOR_SUCCESS, lw=2.2, marker='^', ms=6)
        # Add growth rate labels (color-coded)
        for i, r in enumerate(week_data['Growth Rate']):
            if i>0:
                col = COLOR_SUCCESS if r>0 else COLOR_WARNING
                plt.text(
                    i, 
                    r+0.8 if r>0 else r-1.5, 
                    f"{r}%", 
                    ha='center', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Weekly Customer Traffic - Growth Rate', fontsize=14, weight='bold')
        plt.ylabel('Growth Rate (%)', fontsize=12, c=COLOR_SUCCESS)
        plt.xlabel('Week Date Range', fontsize=12)

    # Monthly growth rate
    elif analysis_type == 'month_growth':
        data = df_ana.groupby(df_ana['Date'].dt.strftime('%Y-%m'))['Customer Traffic'].sum().reset_index()
        data['Growth Rate'] = round(data.iloc[:,1].pct_change()*100, 1)
        plt.plot(data.iloc[:,0], data['Growth Rate'], COLOR_SUCCESS, lw=2.2, marker='^', ms=6)
        # Add growth rate labels (color-coded)
        for i, r in enumerate(data['Growth Rate']):
            if i>0:
                col = COLOR_SUCCESS if r>0 else COLOR_WARNING
                plt.text(
                    i, 
                    r+1.2 if r>0 else r-2.2, 
                    f"{r}%", 
                    ha='center', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Monthly Customer Traffic - Growth Rate', fontsize=14, weight='bold')
        plt.ylabel('Growth Rate (%)', fontsize=12, c=COLOR_SUCCESS)
        plt.xlabel('Year-Month', fontsize=12)

    # Universal styling
    plt.grid(axis='y', alpha=0.2)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()

# Conversion Rate Analysis 
def conversion_rate_analysis(df, analysis_type="daily"):
    """
    Combined conversion rate and average daily sales analysis
    
    analysis_type options:
    - daily: Daily conversion rate only
    - weekly: Weekly conversion rate + avg daily sales amount
    - monthly: Monthly conversion rate + avg daily sales amount
    
    Calculations:
    - Conversion Rate (%) = (Number of Converted Customers / Total Traffic) * 100
    - Avg Daily Sales = Total Sales Amount / Number of Days in Period
    """
    df_ana = df.copy()
    df_ana['Date'] = pd.to_datetime(df_ana['Date'])
    plt.figure(figsize=(11,6), dpi=90)
    color_rate = COLOR_PRIMARY    # Conversion rate - deep blue
    color_sales = COLOR_SUCCESS   # Avg daily sales - burgundy
    fmt_money = FuncFormatter(format_currency) # Currency formatter

    # Daily conversion rate
    if analysis_type == 'daily':
        conv_data = df_ana.groupby('Date').agg(
            Total_Traffic = ('Customer Traffic', 'sum'),
            Converted_Customers = ('Customer Name', 'nunique'),
            Total_Sales = ('Total sales', 'sum')
        ).reset_index()
        conv_data['Conversion_Rate (%)'] = round((conv_data['Converted_Customers'] / conv_data['Total_Traffic']) * 100, 2)

        # Plot conversion rate line
        plt.plot(conv_data['Date'], conv_data['Conversion_Rate (%)'], color=color_rate, lw=2.2, marker='o', ms=5)
        # Add value labels
        for _, row in conv_data.iterrows():
            plt.text(
                row['Date'], 
                row['Conversion_Rate (%)']+0.15, 
                f"{row['Conversion_Rate (%)']}%", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_rate
            )

        plt.title('Daily Customer Conversion Rate', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Conversion Rate (%)', fontsize=12, c=color_rate)

    # Weekly conversion rate + avg daily sales (dual axis)
    elif analysis_type == 'weekly':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        conv_data = df_ana.groupby('Week').agg(
            Total_Traffic = ('Customer Traffic', 'sum'),
            Converted_Customers = ('Customer Name', 'nunique'),
            Total_Sales = ('Total sales', 'sum'),
            Days_in_Week = ('Date', lambda x: (x.max()-x.min()).days + 1)
        ).reset_index()
        # Calculate key metrics
        conv_data['Conversion_Rate (%)'] = round((conv_data['Converted_Customers'] / conv_data['Total_Traffic']) * 100, 2)
        conv_data['Avg_Daily_Sales'] = conv_data['Total_Sales'] / conv_data['Days_in_Week']
        # Format week range
        conv_data['Week Range'] = conv_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])

        # Dual axis setup
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Left axis - conversion rate
        ax1.plot(
            conv_data['Week Range'], 
            conv_data['Conversion_Rate (%)'], 
            color=color_rate, 
            lw=2.2, 
            marker='o', 
            ms=5, 
            label='Conversion Rate'
        )
        for i, val in enumerate(conv_data['Conversion_Rate (%)']):
            ax1.text(
                i, 
                val+0.15, 
                f"{val}%", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_rate
            )
        ax1.set_ylabel('Conversion Rate (%)', fontsize=12, c=color_rate)
        ax1.tick_params(axis='y', colors=color_rate)
        
        # Right axis - avg daily sales (formatted)
        ax2.plot(
            conv_data['Week Range'], 
            conv_data['Avg_Daily_Sales'], 
            color=color_sales, 
            lw=2, 
            marker='^', 
            ms=5, 
            label='Avg Daily Sales', 
            alpha=0.9
        )
        for i, val in enumerate(conv_data['Avg_Daily_Sales']):
            ax2.text(
                i, 
                val+5000, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=8, 
                weight='bold', 
                c=color_sales
            )
        ax2.set_ylabel('Avg Daily Sales Amount', fontsize=12, c=color_sales)
        ax2.tick_params(axis='y', colors=color_sales)
        ax2.yaxis.set_major_formatter(fmt_money)

        plt.title('Weekly Conversion Rate & Avg Daily Sales Amount', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Week Date Range', fontsize=12)

    # Monthly conversion rate + avg daily sales (dual axis)
    elif analysis_type == 'monthly':
        df_ana['Month'] = df_ana['Date'].dt.strftime('%Y-%m')
        conv_data = df_ana.groupby('Month').agg(
            Total_Traffic = ('Customer Traffic', 'sum'),
            Converted_Customers = ('Customer Name', 'nunique'),
            Total_Sales = ('Total sales', 'sum'),
            Days_in_Month = ('Date', lambda x: (x.max()-x.min()).days + 1)
        ).reset_index()
        # Calculate key metrics
        conv_data['Conversion_Rate (%)'] = round((conv_data['Converted_Customers'] / conv_data['Total_Traffic']) * 100, 2)
        conv_data['Avg_Daily_Sales'] = conv_data['Total_Sales'] / conv_data['Days_in_Month']

        # Dual axis setup
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Left axis - conversion rate
        ax1.plot(
            conv_data['Month'], 
            conv_data['Conversion_Rate (%)'], 
            color=color_rate, 
            lw=2.2, 
            marker='o', 
            ms=5, 
            label='Conversion Rate'
        )
        for i, val in enumerate(conv_data['Conversion_Rate (%)']):
            ax1.text(
                i, 
                val+0.15, 
                f"{val}%", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_rate
            )
        ax1.set_ylabel('Conversion Rate (%)', fontsize=12, c=color_rate)
        ax1.tick_params(axis='y', colors=color_rate)
        
        # Right axis - avg daily sales (formatted)
        ax2.plot(
            conv_data['Month'], 
            conv_data['Avg_Daily_Sales'], 
            color=color_sales, 
            lw=2, 
            marker='^', 
            ms=5, 
            label='Avg Daily Sales', 
            alpha=0.9
        )
        for i, val in enumerate(conv_data['Avg_Daily_Sales']):
            ax2.text(
                i, 
                val+8000, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=8, 
                weight='bold', 
                c=color_sales
            )
        ax2.set_ylabel('Avg Daily Sales Amount', fontsize=12, c=color_sales)
        ax2.tick_params(axis='y', colors=color_sales)
        ax2.yaxis.set_major_formatter(fmt_money)

        plt.title('Monthly Conversion Rate & Avg Daily Sales Amount', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Year-Month', fontsize=12)

    else:
        print(f"Parameter error! Only daily/weekly/monthly are supported. Current input: {analysis_type}")
        return

    # Universal styling
    plt.grid(axis='y', alpha=0.2)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()

# Sales Amount Analysis
def sales_amount_analysis(df, analysis_type="daily"):
    """
    Sales amount trend and growth rate analysis
    
    analysis_type options:
    - daily: Daily sales amount trend (line chart)
    - weekly: Weekly sales amount trend (line chart)
    - monthly: Monthly sales amount trend (line chart)
    - week_growth: Weekly growth rate (color-coded bar chart)
    - month_growth: Monthly growth rate (color-coded bar chart)
    
    Features:
    - All values use thousand separators
    - Week axis shows compact date range
    - Growth charts use red for positive, orange for negative growth
    """
    df_ana = df.copy()
    df_ana['Date'] = pd.to_datetime(df_ana['Date'])
    plt.figure(figsize=(10,6), dpi=90)
    color_daily = COLOR_PRIMARY    # Daily sales - deep blue
    color_weekly = COLOR_SECOND    # Weekly sales - magenta
    color_monthly = COLOR_WARNING  # Monthly sales - orange yellow
    color_up = COLOR_SUCCESS       # Positive growth - burgundy
    color_down = COLOR_WARNING     # Negative growth - orange yellow
    fmt_money = FuncFormatter(format_currency) # Currency formatter

    # Daily sales amount trend
    if analysis_type == 'daily':
        sales_data = df_ana.groupby('Date')['Total sales'].sum().reset_index()
        plt.plot(
            sales_data['Date'], 
            sales_data['Total sales'], 
            color=color_daily, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for _, row in sales_data.iterrows():
            plt.text(
                row['Date'], 
                row['Total sales']+8000, 
                f"{int(row['Total sales']):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_daily
            )
        plt.title('Daily Total Sales Amount Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Sales Amount', fontsize=12, c=color_daily)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Weekly sales amount trend (formatted date range)
    elif analysis_type == 'weekly':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        sales_data = df_ana.groupby('Week')['Total sales'].sum().reset_index()
        sales_data['Week Range'] = sales_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        plt.plot(
            sales_data['Week Range'], 
            sales_data['Total sales'], 
            color=color_weekly, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for i, val in enumerate(sales_data['Total sales']):
            plt.text(
                i, 
                val+15000, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_weekly
            )
        plt.title('Weekly Total Sales Amount Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Week Date Range', fontsize=12)
        plt.ylabel('Total Sales Amount', fontsize=12, c=color_weekly)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Monthly sales amount trend
    elif analysis_type == 'monthly':
        df_ana['Month'] = df_ana['Date'].dt.strftime('%Y-%m')
        sales_data = df_ana.groupby('Month')['Total sales'].sum().reset_index()
        plt.plot(
            sales_data['Month'], 
            sales_data['Total sales'], 
            color=color_monthly, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for i, val in enumerate(sales_data['Total sales']):
            plt.text(
                i, 
                val+20000, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_monthly
            )
        plt.title('Monthly Total Sales Amount Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Total Sales Amount', fontsize=12, c=color_monthly)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Weekly growth rate (color-coded bar chart)
    elif analysis_type == 'week_growth':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        sales_data = df_ana.groupby('Week')['Total sales'].sum().reset_index()
        sales_data['Week Range'] = sales_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        # Calculate growth rate (1 decimal place)
        sales_data['Growth_Rate (%)'] = round(sales_data['Total sales'].pct_change() * 100, 1)
        # Color coding: red for positive, orange for negative
        bar_colors = [color_up if x>0 else color_down for x in sales_data['Growth_Rate (%)']]

        # Create bar chart
        x_pos = range(len(sales_data['Week Range']))
        bars = plt.bar(
            x_pos, 
            sales_data['Growth_Rate (%)'], 
            color=bar_colors, 
            width=0.6, 
            alpha=0.9, 
            edgecolor='white', 
            linewidth=1
        )
        # Add growth rate labels (positioned appropriately)
        for i, (val, col) in enumerate(zip(sales_data['Growth_Rate (%)'], bar_colors)):
            if i>0: # Skip first value (no growth rate)
                y_offset = 0.3 if val>0 else -0.8
                plt.text(
                    i, 
                    val+y_offset, 
                    f"{val}%", 
                    ha='center', 
                    va='bottom' if val>0 else 'top', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )

        # Add zero line reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.xticks(x_pos, sales_data['Week Range'])
        plt.title('Weekly Sales Amount - Growth Rate', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Week Date Range', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12, c=color_up)

    # Monthly growth rate (color-coded bar chart)
    elif analysis_type == 'month_growth':
        df_ana['Month'] = df_ana['Date'].dt.strftime('%Y-%m')
        sales_data = df_ana.groupby('Month')['Total sales'].sum().reset_index()
        # Calculate growth rate (1 decimal place)
        sales_data['Growth_Rate (%)'] = round(sales_data['Total sales'].pct_change() * 100, 1)
        # Color coding: red for positive, orange for negative
        bar_colors = [color_up if x>0 else color_down for x in sales_data['Growth_Rate (%)']]

        # Create bar chart
        x_pos = range(len(sales_data['Month']))
        bars = plt.bar(
            x_pos, 
            sales_data['Growth_Rate (%)'], 
            color=bar_colors, 
            width=0.6, 
            alpha=0.9, 
            edgecolor='white', 
            linewidth=1
        )
        # Add growth rate labels (positioned appropriately)
        for i, (val, col) in enumerate(zip(sales_data['Growth_Rate (%)'], bar_colors)):
            if i>0: # Skip first value (no growth rate)
                y_offset = 0.5 if val>0 else -1.2
                plt.text(
                    i, 
                    val+y_offset, 
                    f"{val}%", 
                    ha='center', 
                    va='bottom' if val>0 else 'top', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )

        # Add zero line reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.xticks(x_pos, sales_data['Month'])
        plt.title('Monthly Sales Amount - Growth Rate', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12, c=color_up)

    else:
        print(f"Parameter error! Only daily/weekly/monthly/week_growth/month_growth are supported. Current input: {analysis_type}")
        return

    # Universal styling
    plt.grid(axis='y', alpha=0.2)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()

# Sales Quantity Analysis 
def sales_quantity_analysis(df, analysis_type="daily"):
    df_ana = df.copy()
    df_ana['Date'] = pd.to_datetime(df_ana['Date'])
    plt.figure(figsize=(10,6), dpi=90)
    color_daily = COLOR_PRIMARY    # Daily sales - deep blue
    color_weekly = COLOR_SECOND    # Weekly sales - magenta
    color_monthly = COLOR_WARNING  # Monthly sales - orange yellow
    color_up = COLOR_SUCCESS       # Positive growth - burgundy
    color_down = COLOR_WARNING     # Negative growth - orange yellow
    fmt_money = FuncFormatter(format_currency) # Number formatter

    # Daily sales quantity trend
    if analysis_type == 'daily':
        sales_data = df_ana.groupby('Date')['Sales'].sum().reset_index()
        plt.plot(
            sales_data['Date'], 
            sales_data['Sales'], 
            color=color_daily, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for _, row in sales_data.iterrows():
            plt.text(
                row['Date'], 
                row['Sales']+0.5, 
                f"{int(row['Sales']):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_daily
            )
        plt.title('Daily Total Sales Quantity Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Sales Quantity', fontsize=12, c=color_daily)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Weekly sales quantity trend (formatted date range)
    elif analysis_type == 'weekly':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        sales_data = df_ana.groupby('Week')['Sales'].sum().reset_index()
        sales_data['Week Range'] = sales_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        plt.plot(
            sales_data['Week Range'], 
            sales_data['Sales'], 
            color=color_weekly, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for i, val in enumerate(sales_data['Sales']):
            plt.text(
                i, 
                val+1, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_weekly
            )
        plt.title('Weekly Total Sales Quantity Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Week Date Range', fontsize=12)
        plt.ylabel('Total Sales Quantity', fontsize=12, c=color_weekly)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Monthly sales quantity trend
    elif analysis_type == 'monthly':
        df_ana['Month'] = df_ana['Date'].dt.strftime('%Y-%m')
        sales_data = df_ana.groupby('Month')['Sales'].sum().reset_index()
        plt.plot(
            sales_data['Month'], 
            sales_data['Sales'], 
            color=color_monthly, 
            lw=2.2, 
            marker='o', 
            ms=5
        )
        # Add formatted value labels
        for i, val in enumerate(sales_data['Sales']):
            plt.text(
                i, 
                val+1.5, 
                f"{int(val):,}", 
                ha='center', 
                va='bottom', 
                fontsize=9, 
                weight='bold', 
                c=color_monthly
            )
        plt.title('Monthly Total Sales Quantity Trend', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Total Sales Quantity', fontsize=12, c=color_monthly)
        plt.gca().yaxis.set_major_formatter(fmt_money)

    # Weekly growth rate (color-coded bar chart)
    elif analysis_type == 'week_growth':
        df_ana['Week'] = df_ana['Date'].dt.to_period('W')
        sales_data = df_ana.groupby('Week')['Sales'].sum().reset_index()
        sales_data['Week Range'] = sales_data['Week'].astype(str).apply(lambda x: x.split('/')[0][5:] + '-' + x.split('/')[1][5:])
        # Calculate growth rate (1 decimal place)
        sales_data['Growth_Rate (%)'] = round(sales_data['Sales'].pct_change() * 100, 1)
        # Color coding: red for positive, orange for negative
        bar_colors = [color_up if x>0 else color_down for x in sales_data['Growth_Rate (%)']]

        # Create bar chart
        x_pos = range(len(sales_data['Week Range']))
        bars = plt.bar(
            x_pos, 
            sales_data['Growth_Rate (%)'], 
            color=bar_colors, 
            width=0.6, 
            alpha=0.9, 
            edgecolor='white', 
            linewidth=1
        )
        # Add growth rate labels (positioned appropriately)
        for i, (val, col) in enumerate(zip(sales_data['Growth_Rate (%)'], bar_colors)):
            if i>0: # Skip first value (no growth rate)
                y_offset = 0.3 if val>0 else -0.8
                plt.text(
                    i, 
                    val+y_offset, 
                    f"{val}%", 
                    ha='center', 
                    va='bottom' if val>0 else 'top', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )

        # Add zero line reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.xticks(x_pos, sales_data['Week Range'])
        plt.title('Weekly Sales Quantity - Growth Rate', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Week Date Range', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12, c=color_up)

    # Monthly growth rate (color-coded bar chart)
    elif analysis_type == 'month_growth':
        df_ana['Month'] = df_ana['Date'].dt.strftime('%Y-%m')
        sales_data = df_ana.groupby('Month')['Sales'].sum().reset_index()
        # Calculate growth rate (1 decimal place)
        sales_data['Growth_Rate (%)'] = round(sales_data['Sales'].pct_change() * 100, 1)
        # Color coding: red for positive, orange for negative
        bar_colors = [color_up if x>0 else color_down for x in sales_data['Growth_Rate (%)']]

        # Create bar chart
        x_pos = range(len(sales_data['Month']))
        bars = plt.bar(
            x_pos, 
            sales_data['Growth_Rate (%)'], 
            color=bar_colors, 
            width=0.6, 
            alpha=0.9, 
            edgecolor='white', 
            linewidth=1
        )
        # Add growth rate labels (positioned appropriately)
        for i, (val, col) in enumerate(zip(sales_data['Growth_Rate (%)'], bar_colors)):
            if i>0: # Skip first value (no growth rate)
                y_offset = 0.5 if val>0 else -1.2
                plt.text(
                    i, 
                    val+y_offset, 
                    f"{val}%", 
                    ha='center', 
                    va='bottom' if val>0 else 'top', 
                    fontsize=9, 
                    weight='bold', 
                    c=col
                )

        # Add zero line reference
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.6, linewidth=1)
        plt.xticks(x_pos, sales_data['Month'])
        plt.title('Monthly Sales Quantity - Growth Rate', fontsize=14, weight='bold', pad=20)
        plt.xlabel('Year-Month', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12, c=color_up)

    else:
        print(f"Parameter error! Only daily/weekly/monthly/week_growth/month_growth are supported. Current input: {analysis_type}")
        return

    # Universal styling
    plt.grid(axis='y', alpha=0.2)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
