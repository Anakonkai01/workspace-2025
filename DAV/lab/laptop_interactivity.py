# Import required libraries
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
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

# 3. Classify CPU
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

# 4. Create Price Segment
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
                               
app.layout = html.Div(children=[
    html.H1('Laptop Performance Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    html.Div([
        "Select CPU Category: ", 
        dcc.Dropdown(
            id='input-cpu',
            options=[
                {'label': 'All CPUs', 'value': 'ALL'},
                {'label': 'Intel Core i9', 'value': 'Intel Core i9'},
                {'label': 'Intel Core i7', 'value': 'Intel Core i7'},
                {'label': 'Intel Core i5', 'value': 'Intel Core i5'},
                {'label': 'Intel Core i3', 'value': 'Intel Core i3'},
                {'label': 'AMD Ryzen 9', 'value': 'AMD Ryzen 9'},
                {'label': 'AMD Ryzen 7', 'value': 'AMD Ryzen 7'},
                {'label': 'AMD Ryzen 5', 'value': 'AMD Ryzen 5'},
                {'label': 'Other', 'value': 'Other'}
            ],
            value='ALL',
            style={'height':'50px', 'font-size': 20}
        )
    ], style={'font-size': 30, 'width': '50%', 'margin': 'auto'}),
    
    html.Br(),
    html.Br(),
    html.Div(dcc.Graph(id='line-plot')),
])


# Add callback decorator
@app.callback(
    Output(component_id='line-plot', component_property='figure'),
    Input(component_id='input-cpu', component_property='value')
)
# Add computation to callback function and return graph
def get_graph(selected_cpu):
    # Filter data based on CPU selection
    if selected_cpu == 'ALL':
        df = laptop_data
    else:
        df = laptop_data[laptop_data['CPU_Category'] == selected_cpu]
    
    # Group the data by RAM and compute average price
    line_data = df.groupby('RAM_GB')['Price_VND'].mean().reset_index()
    
    # Sort by RAM for better visualization
    line_data = line_data.sort_values('RAM_GB')
    
    # Create the figure
    fig = go.Figure(data=go.Scatter(
        x=line_data['RAM_GB'], 
        y=line_data['Price_VND'], 
        mode='lines+markers', 
        marker=dict(color='green', size=10),
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f'RAM vs Average Laptop Price - {selected_cpu}',
        xaxis_title='RAM (GB)',
        yaxis_title='Average Price (VND)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
