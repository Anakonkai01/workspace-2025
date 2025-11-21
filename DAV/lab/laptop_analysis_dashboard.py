# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import re
import numpy as np

# Read the laptop data into pandas dataframe
laptop_data = pd.read_csv('laptops_asus_data_cellphones_full_v2.csv')

# Preprocess data
# 1. Clean Price column
laptop_data['Price_VND'] = (
    laptop_data['Price']
    .astype(str)
    .str.replace('đ', '', regex=False)
    .str.replace('.', '', regex=False)
    .str.strip()
)
laptop_data['Price_VND'] = pd.to_numeric(laptop_data['Price_VND'], errors='coerce')

# 2. Clean RAM column
laptop_data['RAM_GB'] = (
    laptop_data['RAM']
    .astype(str)
    .str.replace('GB', '', regex=False)
    .str.strip()
)
laptop_data['RAM_GB'] = pd.to_numeric(laptop_data['RAM_GB'], errors='coerce').astype('Int64')

# 3. Clean Storage column
def clean_storage(storage_str):
    if pd.isna(storage_str) or storage_str.upper() == 'NAN':
        return np.nan
    storage_str = storage_str.upper()
    tb_match = re.search(r'(\d+)\s*TB', storage_str)
    if tb_match:
        return float(tb_match.group(1)) * 1024
    gb_match = re.search(r'(\d+)\s*GB', storage_str)
    if gb_match:
        return float(gb_match.group(1))
    return np.nan

laptop_data['Storage_GB'] = laptop_data['Storage'].astype(str).apply(clean_storage)

# 4. Classify CPU
def classify_cpu(cpu_str):
    if pd.isna(cpu_str):
        return 'Unknown'
    cpu_str = str(cpu_str).lower()
    if 'core i9' in cpu_str or 'core 9' in cpu_str:
        return 'Intel Core i9'
    elif 'core i7' in cpu_str or 'core 7' in cpu_str:
        return 'Intel Core i7'
    elif 'core i5' in cpu_str or 'core 5' in cpu_str:
        return 'Intel Core i5'
    elif 'core i3' in cpu_str or 'core 3' in cpu_str:
        return 'Intel Core i3'
    elif 'ryzen 9' in cpu_str:
        return 'AMD Ryzen 9'
    elif 'ryzen 7' in cpu_str:
        return 'AMD Ryzen 7'
    elif 'ryzen 5' in cpu_str:
        return 'AMD Ryzen 5'
    else:
        return 'Other'

laptop_data['CPU_Category'] = laptop_data['CPU'].apply(classify_cpu)

# 5. Classify GPU
def classify_gpu(gpu_str):
    if pd.isna(gpu_str):
        return 'Unknown'
    gpu_str = str(gpu_str).lower()
    if 'rtx 4090' in gpu_str:
        return 'RTX 4090'
    elif 'rtx 4080' in gpu_str:
        return 'RTX 4080'
    elif 'rtx 4070' in gpu_str:
        return 'RTX 4070'
    elif 'rtx 4060' in gpu_str:
        return 'RTX 4060'
    elif 'rtx 4050' in gpu_str:
        return 'RTX 4050'
    elif 'rtx 3050' in gpu_str:
        return 'RTX 3050'
    elif 'rtx' in gpu_str:
        return 'RTX Other'
    elif 'gtx' in gpu_str:
        return 'GTX'
    elif 'radeon' in gpu_str:
        return 'AMD Radeon'
    elif 'intel' in gpu_str or 'iris' in gpu_str or 'uhd' in gpu_str:
        return 'Intel Integrated'
    else:
        return 'Other'

laptop_data['GPU_Category'] = laptop_data['GPU'].apply(classify_gpu)

# 6. Create Price Segment
def price_segment(price):
    if pd.isna(price):
        return 'Unknown'
    elif price < 15000000:
        return 'Budget (<15M)'
    elif price < 25000000:
        return 'Mid-range (15-25M)'
    elif price < 40000000:
        return 'High-end (25-40M)'
    else:
        return 'Premium (>40M)'

laptop_data['Price_Segment'] = laptop_data['Price_VND'].apply(price_segment)

# Remove rows with missing critical data
laptop_data = laptop_data.dropna(subset=['Price_VND', 'RAM_GB'])

# Create a dash application
app = dash.Dash(__name__)

# Build dash app layout
app.layout = html.Div(children=[
    html.H1('Laptop Analysis Dashboard', 
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    html.Div([
        "Select Price Segment: ", 
        dcc.Dropdown(
            id='input-segment',
            options=[
                {'label': 'All Segments', 'value': 'ALL'},
                {'label': 'Budget (<15M)', 'value': 'Budget (<15M)'},
                {'label': 'Mid-range (15-25M)', 'value': 'Mid-range (15-25M)'},
                {'label': 'High-end (25-40M)', 'value': 'High-end (25-40M)'},
                {'label': 'Premium (>40M)', 'value': 'Premium (>40M)'}
            ],
            value='ALL',
            style={'height':'50px', 'font-size': 20}
        )
    ], style={'font-size': 30, 'width': '50%', 'margin': 'auto'}),
    
    html.Br(),
    html.Br(),
    
    # Segment 1: CPU and GPU Distribution
    html.Div([
        html.Div(dcc.Graph(id='cpu-plot')),
        html.Div(dcc.Graph(id='gpu-plot'))
    ], style={'display': 'flex'}),
    
    # Segment 2: Price Analysis
    html.Div([
        html.Div(dcc.Graph(id='ram-price-plot')),
        html.Div(dcc.Graph(id='storage-price-plot'))
    ], style={'display': 'flex'}),
    
    # Segment 3: Price Distribution
    html.Div(dcc.Graph(id='price-distribution'), style={'width':'70%', 'margin': 'auto'})
])

""" Compute_info function description

This function takes in laptop data and selected price segment as an input and performs computation for creating charts and plots.

Arguments:
    laptop_data: Input laptop data.
    selected_segment: Input price segment for which computation needs to be performed.
    
Returns:
    Computed dataframes for CPU count, GPU count, RAM-Price avg, Storage-Price avg, and price distribution.
"""
def compute_info(laptop_data, selected_segment):
    # Select data
    if selected_segment == 'ALL':
        df = laptop_data
    else:
        df = laptop_data[laptop_data['Price_Segment'] == selected_segment]
    
    # Compute statistics
    cpu_count = df.groupby('CPU_Category').size().reset_index(name='Count')
    gpu_count = df.groupby('GPU_Category').size().reset_index(name='Count')
    ram_price = df.groupby('RAM_GB')['Price_VND'].mean().reset_index()
    storage_price = df[df['Storage_GB'].notna()].groupby('Storage_GB')['Price_VND'].mean().reset_index()
    price_dist = df['Price_VND']
    
    return cpu_count, gpu_count, ram_price, storage_price, price_dist

"""Callback Function

Function that returns figures using the provided input price segment.

Arguments:
    selected_segment: Input price segment provided by the user.
    
Returns:
    List of figures computed using the provided helper function `compute_info`.
"""
# Callback decorator
@app.callback([
    Output(component_id='cpu-plot', component_property='figure'),
    Output(component_id='gpu-plot', component_property='figure'),
    Output(component_id='ram-price-plot', component_property='figure'),
    Output(component_id='storage-price-plot', component_property='figure'),
    Output(component_id='price-distribution', component_property='figure')
],
    Input(component_id='input-segment', component_property='value')
)
# Computation to callback function and return graph
def get_graph(selected_segment):
    
    # Compute required information for creating graph from the data
    cpu_count, gpu_count, ram_price, storage_price, price_dist = compute_info(laptop_data, selected_segment)
    
    # Bar chart for CPU distribution
    cpu_fig = px.bar(cpu_count, x='CPU_Category', y='Count', 
                     title=f'Laptop Count by CPU Category - {selected_segment}',
                     color='Count',
                     color_continuous_scale='Blues')
    cpu_fig.update_layout(xaxis_tickangle=-45)
    
    # Bar chart for GPU distribution
    gpu_fig = px.bar(gpu_count, x='GPU_Category', y='Count', 
                     title=f'Laptop Count by GPU Category - {selected_segment}',
                     color='Count',
                     color_continuous_scale='Greens')
    gpu_fig.update_layout(xaxis_tickangle=-45)
    
    # Line plot for RAM vs Average Price
    ram_price = ram_price.sort_values('RAM_GB')
    ram_fig = px.line(ram_price, x='RAM_GB', y='Price_VND',
                      title=f'Average Price by RAM - {selected_segment}',
                      markers=True)
    ram_fig.update_traces(line_color='red', marker=dict(size=10))
    
    # Line plot for Storage vs Average Price
    storage_price = storage_price.sort_values('Storage_GB')
    storage_fig = px.line(storage_price, x='Storage_GB', y='Price_VND',
                          title=f'Average Price by Storage - {selected_segment}',
                          markers=True)
    storage_fig.update_traces(line_color='purple', marker=dict(size=10))
    
    # Histogram for Price Distribution
    price_fig = px.histogram(price_dist, x=price_dist, nbins=30,
                             title=f'Price Distribution - {selected_segment}',
                             labels={'x': 'Price (VND)', 'count': 'Number of Laptops'})
    price_fig.update_traces(marker_color='orange')
    
    return [cpu_fig, gpu_fig, ram_fig, storage_fig, price_fig]

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8051)
