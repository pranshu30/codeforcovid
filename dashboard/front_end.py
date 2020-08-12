import os
import pathlib
import re

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

from data.external_data import CovidData
from trans_opt.opt import TransOpt


# Initialize app

app1 = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.SLATE]
)
server = app1.server

# Load data

APP_PATH = str(pathlib.Path(__file__).parent.resolve())
data = CovidData()
data.get_county_cases(max_number=250)
data.get_county_coordinates()




TIME = list(range(0,data.df_cases.shape[0] - 1))

CASES_PER_DAY = 'approximate_cases_per_day'
AGGREGATE_PER_DAY = 'agg_per_day'
DC_PER_DAY = 'dc_per_day'
PRODUCTION_PER_DAY = 'prod_per_day'









DEFAULT_OPACITY = 0.8

mapbox_access_token = "pk.eyJ1Ijoic2FtY294MjM2IiwiYSI6ImNqZmpzdzZ4dDBkcnczM3BtanVjZHg3b2oifQ.NNfmYvzRF2F7FRFKBiTfUw"
mapbox_style = "mapbox://styles/plotlymapbox/cjvprkf3t1kns1cqjxuxmwixz"

# App layout

app1.layout = dbc.Container(
    id="root",
    fluid=True,
    children=[
        dbc.Row(
            id="header",
            align='start',
            children=[
                html.H4(children="COVID19 Surge Predictor & Optimized Distribution of PPE"),
            ]
        ),
        dbc.Row(
            id="controls",
            align='start',
            children=[dbc.Col(
                id='slider-col',
                children=[
                    html.Div(
                                id="slider-container",
                                children=[
                                    dbc.Button("Open modal", id="open"),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader("Header"),
                                            dbc.ModalBody(
                                                [
                                                    dbc.Input(id=data.fips_to_pretty[dc].split('<br>')[-1],
                                                            placeholder=f'Enter Starting amount at {" ".join(data.fips_to_pretty[dc].split("<br>")[0:2])}',
                                                                type='number',
                                                                max=20000,
                                                                min=0)
                                                    for dc in data.dc_geo.index.unique().tolist()
                                                ] + [dbc.Button(id='opt-button', children='Optimize')]
                                                ),
                                            dbc.ModalFooter(
                                                    dbc.Button("Close", id="close", className="ml-auto")
                                            ),
                                        ],
                                        id="modal",
                                    ),
                                    dcc.RangeSlider(
                                        id="years-slider",
                                        min=min(TIME),
                                        max=max(TIME),
                                        value=[TIME[0], TIME[10]],
                                        marks={
                                            str(time.strftime('%m/%d/%Y')): {
                                                "label": time.strftime('%m/%d/%Y'),
                                                # "style": {"color": "#7fafdf"},
                                            }
                                            for i, time in enumerate(data.df_cases.index.tolist()) if i%10 == 0
                                        },
                                    ),
                                    html.P(
                                        "Number of cases",
                                        id="heatmap-title",
                                    ),
                                ],
                            ),
                ]
            )
            ]
        ),
        dbc.Row(
            id="app-container",
            align='end',
            children=[
                dbc.Col(
                    id="left-column",
                    align='center',
                    style={'background':'primary'},
                    children=[
                        html.Div(
                            id="heatmap-container",
                            children=[
                                dcc.Graph(
                                    id="county-choropleth",
                                    figure=dict(
                                        layout=dict(
                                            mapbox=dict(
                                                layers=[],
                                                accesstoken=mapbox_access_token,
                                                style=mapbox_style,
                                                center=dict(
                                                    lat=38.72490, lon=-95.61446
                                                ),
                                                pitch=0,
                                                zoom=3.5,
                                            ),
                                            autosize=True,
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    ]
                ),
                dbc.Col(
                    id="graph-container",
                    align='center',
                    style={'background':'secondary'},
                    children=[
                        html.P(id="chart-selector", children="Select chart:"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "Approximate Cases Per Day",
                                    "value": CASES_PER_DAY,
                                },
                                {
                                    'label': 'Cumulative Cases Per Day',
                                    'value': AGGREGATE_PER_DAY
                                },
                                {
                                    'label': 'Number of cases Stored in DC per day',
                                    'value': DC_PER_DAY
                                },
                                {
                                    'label': 'Number of cases produced per day',
                                    'value': PRODUCTION_PER_DAY
                                }
                            ],
                            value=CASES_PER_DAY,
                            id="chart-dropdown",
                        ),
                        dcc.Graph(
                            id="selected-data",
                            figure=dict(
                                data=[dict(x=0, y=0)],
                                layout=dict(
                                    paper_bgcolor="#F4F4F8",
                                    plot_bgcolor="#F4F4F8",
                                    autofill=True,
                                    margin=dict(t=75, r=50, b=100, l=50),
                                ),
                            ),
                        ),
                        dcc.Store(id='opt-store')
                    ]
                ),
            ]
        ),
    ]
)


@app1.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app1.callback(Output('opt-store', 'data'),
               [Input('opt-button', 'n_clicks')],
               [State('years-slider', 'value'),
                State('47137', 'value'),
                State('56027', 'value'),
                State('41033', 'value'),
                State('39109', 'value')])
def opt_store(n_clicks, days, n1, n2, n3, n4):
        trans_model = TransOpt(data)
        
        n1=10000 if n1 is None else n1
        n2=10 if n2 is None else n2
        n3=10 if n3 is None else n3
        n4=10 if n4 is None else n4
        
        trans_model.create_model(data.df_cases, start_day = days[0], time_horizon=days[1] - days[0],
                                 n1=n1,
                                 n2=n2,
                                 n3=n3,
                                 n4=n4)
        trans_model.solve()
        dc_data = dict()
        for dc in trans_model.model.dcs:
            dc_data[dc] = dict(x = [t for t in trans_model.model.t],
                        y = [trans_model.model.cases_held[dc, t].value for t in trans_model.model.t],
                        name=data.fips_to_pretty[dc])
        prod_data = dict()
        for prod in trans_model.model.production_nodes:
            prod_data[prod] = dict(x = [t for t in trans_model.model.t],
            y = [trans_model.model.cases_produced[prod, t].value for t in trans_model.model.t],
            name=data.fips_to_pretty[prod])
        dc_dem = dict()
        for dc in trans_model.model.dcs:
            dc_to_dem = dict()
            for dem_node in trans_model.model.demand_nodes:
                sum_dc_dem = sum(trans_model.model.cases_dc_dem[dc,dem_node,t].value for t in trans_model.model.t)
                if sum_dc_dem > 0.0:
                    dc_to_dem[dem_node] = sum_dc_dem
            dc_dem[dc] = dc_to_dem
        print(dc_dem)
            
        return dict(prod_data = prod_data, dc_data = dc_data, dc_to_demand = dc_dem)
            

@app1.callback(
    Output("county-choropleth", "figure"),
    [Input("years-slider", "value"),
     Input('county-choropleth', 'hoverData')],
    [State("county-choropleth", "figure"),
     State("opt-store", "data")],
)
def display_map(day, hover_data, figure, opt_data):
    
    fig = go.Figure()
    
    if hover_data is not None:
        if 'Dist' in hover_data['points'][0]['text']:
            dc = hover_data['points'][0]['text'].split('<br>')[2].split(' ')[0]
            for k,v in opt_data['dc_to_demand'][dc].items():
                try:
                    fig.add_trace(go.Scattermapbox(
                        lat = [data.dc_geo.loc[dc,'latitude'], data.df_geo.loc[k,'latitude']],
                        lon=[data.dc_geo.loc[dc,'longitude'], data.df_geo.loc[k,'longitude']],
                        mode = 'lines',
                        marker=go.scattermapbox.Marker(
                            size=v,
                            color='rgb(255,255,255)',
                            opacity=0.2
                        ),
                        text=None,
            ))
                except KeyError:
                    print('missing keys or something along those lines')
    
    fig.add_trace(go.Scattermapbox(
        lat = data.df_geo['latitude'],
        lon=data.df_geo['longitude'],
        mode = 'markers',
        marker=go.scattermapbox.Marker(
            size=[data.df_cases.loc[:,column].iloc[day[0]:day[1]].sum()**0.3 for column in data.df_geo.index.values.tolist()],
            color='rgb(255, 0, 0)',
            opacity=0.5
        ),
        text=[data.fips_to_pretty[c] for c in data.df_geo.index.values.tolist()],
        # hoverinfo=[data.df_cases.loc[:,column].iloc[0] for column in data.df_geo.index.values.tolist()]
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat = data.dc_geo['latitude'],
        lon=data.dc_geo['longitude'],
        mode = 'markers',
        marker=go.scattermapbox.Marker(
            symbol='triangle',
            size=20,
            color='rgb(152,251,152)'
        ),
        text=[str(data.fips_to_pretty[dc]) + ' Distribution Center' for dc in data.dc_geo.index.values.tolist()],
        # hoverinfo=[data.df_cases.loc[:,column].iloc[0] for column in data.df_geo.index.values.tolist()]
    ))
    
    fig.add_trace(go.Scattermapbox(
        lat = data.production_geo['latitude'],
        lon=data.production_geo['longitude'],
        mode = 'markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='rgb(0,128,0)',
            symbol='square'
        ),
        text=[str(data.fips_to_pretty[prod]) + ' Production Plant' for prod in data.production_geo.index.values.tolist()],
        # hoverinfo=[data.df_cases.loc[:,column].iloc[0] for column in data.df_geo.index.values.tolist()]
    ))
    
    fig.update_layout(
    title='#CodeForCOVID19',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#2cfec1"),
    margin=dict(t=25, r=10, b=10, l=10),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=figure["layout"]["mapbox"]["center"]["lat"],
            lon=figure["layout"]["mapbox"]["center"]["lon"]
        ),
        pitch=0,
        zoom=figure["layout"]["mapbox"]["zoom"],
        style='dark'
    ),
)
    
    return fig


@app1.callback(Output("heatmap-title", "children"), [Input("years-slider", "value")])
def update_map_title(day):
    return "Daily case count {0} - {1}".format(
        str(data.df_cases.iloc[day[0]].name.strftime('%m/%d/%Y')),
        str(data.df_cases.iloc[day[1]].name.strftime('%m/%d/%Y'))
    )


@app1.callback(
    Output("selected-data", "figure"),
    [
        Input("county-choropleth", "selectedData"),
        Input("chart-dropdown", "value"),
        Input("years-slider", "value"),
        Input('opt-store', 'data')
    ],
)
def display_selected_data(selectedData, chart_dropdown, year, opt_data):
    fig = go.Figure(layout = dict(
                title="Click-drag on the map to select counties",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#2cfec1"),
                margin=dict(t=75, r=50, b=20, l=10),
            ))
    

    
    if chart_dropdown in [AGGREGATE_PER_DAY, CASES_PER_DAY]:
        
        if selectedData is None:
            df = data.df_cases.iloc[year[0]:year[1]].copy()
            # df['total_cases'] = df.sum(axis=1)
            # df = df[['total_cases']]
        else:
            selected_counties = [point['text'].split('<br>')[-1] for point in selectedData['points'] if 'Dist' not in point['text']]
            df = data.df_cases[selected_counties].iloc[year[0]:year[1]].copy()
            
        if chart_dropdown == AGGREGATE_PER_DAY:
            df = df.cumsum()
    
        for col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x = df.index.tolist(),
                    y=df[col].values,
                    name=data.fips_to_pretty[col],
                    mode='lines+markers'
                )
            )
    
    
    if chart_dropdown == DC_PER_DAY:
        for dc, v in opt_data['dc_data'].items():
            fig.add_trace(
                go.Scatter(
                    x = v['x'],
                    y = v['y'],
                    name=data.fips_to_pretty[dc],
                    mode='lines+markers'
                )
            )
            
    if chart_dropdown == PRODUCTION_PER_DAY:
        for prod, v in opt_data['prod_data'].items():
            fig.add_trace(
                go.Scatter(
                    x = v['x'],
                    y = v['y'],
                    name=data.fips_to_pretty[prod],
                    mode='lines+markers'
                )
            )
        
            
    return fig


if __name__ == "__main__":
    app1.run_server(debug=True)
