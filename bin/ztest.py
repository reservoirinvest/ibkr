# ztest.py
import asyncio
from ib_insync import *
util.patchAsyncio()

import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.platform.asyncio import AsyncIOMainLoop

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

ib = IB()
ib.connect('127.0.0.1', 3000, clientId=0)

app.layout = html.Div([
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Coke', 'value': 'COKE'},
            {'label': 'Tesla', 'value': 'TSLA'},
            {'label': 'Apple', 'value': 'AAPL'}
        ],
        value='COKE'
    ),
    dcc.Graph(id='my-graph')
], style={'width': '500'})

@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    contract = Stock(selected_dropdown_value, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='365 D',
            barSizeSetting='1 day', whatToShow='TRADES', useRTH=False,
            formatDate=1, keepUpToDate=False)
    df = util.df(bars)
    return {
        'data': [{
            'x': df.date,
            'y': df.close
        }],
        'layout': {'margin': {'l': 40, 'r': 0, 't': 20, 'b': 30}}
    }


if __name__ == "__main__":
    AsyncIOMainLoop().install()
    http_server = HTTPServer(WSGIContainer(server))
    http_server.listen(8000)
    asyncio.get_event_loop().run_forever()

#_____________________________________

