#imports
#-----general libraries-----
import pandas as pd
import xlwings as xw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#-----sklearn-libraries-----
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
#-----others-----------------   
import base64
from io import BytesIO
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
import shap
from collections import Counter
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from statistics import mean
#Importing data, password-protected
import pandas as pd
import xlwings as xw
import sys
from sklearn.metrics import roc_curve, auc
# Setup Dash
import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash_table.Format import Format, Group, Scheme, Symbol
import sys
import shap
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/Zana/anaconda3/envs/PIPRA/Work_pipra/app/')
from settings import config, about

from python.data import Data
from python.model import Model
from python.result import Result
import plotly.graph_objects as go

MODELS = {'Distribution age/gender': 1,
          'AUC by gender': 2}

#images
radar_logo = 'C:/Users/Zana/anaconda3/envs/PIPRA/Work_pipra/app/assets/my-image.png'
#use this for img load = encode_logo = base64.b64encode(open(radar_logo, 'rb').read()).decode('ascii')
encode_logo = base64.b64encode(open(radar_logo, 'rb').read())
# Read data
data = Data()
df, f_list = data.get_data()
X = df.iloc[:,1:]
y = df.iloc[:,0:1]
    
X_trainUC, X_testUC, y_trainUC, y_testUC = train_test_split(X, y, test_size=0.2, random_state=19)
x_new = X_testUC[X_testUC.index == 549.0]
dfSun = pd.read_csv('df_sunburst_MF.csv')

def createSunburst(df):
    figure = px.sunburst(df, path=["sex (0 male, 1 female)","pod","# comorbidities","BMI_category"], color = 'pod')
    figure.update_layout(width=700, height=700,uniformtext=dict(minsize=12))
    figure.update_traces(
        go.Sunburst(
        insidetextorientation='radial'))
    return figure




# App Instance
app = dash.Dash(name=config.name, assets_folder=config.root+"/application/static", external_stylesheets=[dbc.themes.LUX, config.fontawesome])
app.title = config.name

#Filtering reading the cleaned csv file.
df_ucnoc = pd.read_csv('df_nocurgn.csv')
#Decide how to proceed with that

# Navbar
navbar = dbc.Nav(className="nav nav-pills", children=[
    ## logo/home
    dbc.NavItem(html.Img(src=app.get_asset_url("logo.png"), height="40px")),
    ## about
    dbc.NavItem(html.Div([
        dbc.NavLink("About", href="/", id="about-popover", active=False),
        dbc.Popover(id="about", is_open=False, target="about-popover", children=[
            dbc.PopoverHeader("How it works"), dbc.PopoverBody(about.txt)
        ])
    ]))
])
# Input

x_newDF = pd.DataFrame(columns=['value','feature','shape_original'])
# App Layout

def drawText():
    return html.Div([
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.H2("Text"),
                ], style={'textAlign': 'center'}) 
            ]),style ={'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px',
                    'background-color': '#F5F8FB',
                    'margin': '10px',
                    'padding': '15px',
                    'position': 'relative'
            }
        ),
    ])


app.layout = dbc.Container(fluid=True, children=[
    
    dbc.Card(
    dbc.CardBody([
    ## Top
    html.H1(config.name, id="nav-pills"),
    navbar,
    html.Br(),html.Br(),html.Br(),
    html.Div([html.H2('General Patient Data Overview',
                style={'marginLeft': 20, 'color': 'white'})],
        style={
            'backgroundColor': '#C3CDD6',
            'padding': '10px 5px',
            'box-shadow': '2px 2px 2px lightgrey',
            'border-radius': '5px' }),
    
    dbc.Row([

    ## Body
    
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
         html.Div([
                html.P("Pre-defined Filtering Options:"),
                dcc.Dropdown(
                id='model-name',
                options=[{'label': x, 'value': x} for x in MODELS],
                value='AUC by gender',
                clearable=False
                            ),           
         dcc.Graph(id="graph"),
        ]),
        html.Br(),
        ]),style ={'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px',
                    'border': '10px solid rgba(255,255,255,.5)',
                    'background-color': '#F5F8FB',
                    'margin': '10px',
                    'padding': '15px',
                    'position': 'relative'
            })
        ], width = 6)

    ,
        
        
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
           html.Div(className="four columns pretty_container", children=[
            html.Label('Select ASA-Status'),
            dcc.Dropdown(id='days',
                         placeholder='Filter after ASA status',
                         options=[{'label': 'ALL', 'value': 10.0},
                                  {'label': 'ASA 1', 'value': 1.0},
                                  {'label': 'ASA 2', 'value': 2.0},
                                  {'label': 'ASA 3', 'value': 3.0},
                                  {'label': 'ASA 4', 'value': 4.0},
                                  {'label': 'ASA 5', 'value': 5.0},
                                  {'label': 'ASA 6', 'value': 6.0}],
                         value=[]),
        ]),
        html.Div("Sunburst Chart for current patient data",
                          style={'font-weight': 'bold', 'font-size': 20}),
        html.Center(
            dcc.Graph(id='sunnyburst')
                              #figure=createSunburst(dfSun))
        ),
        ]),
        style ={'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px',
                    'border': '10px solid rgba(255,255,255,.5)',
                    'background-color': '#F5F8FB',
                    'margin': '20px',
                    'padding': '15px',
                    'position': 'relative'
            })
        ], width = 6)

        ],justify="center")
,
    html.Br(),html.Br(),html.Br(),html.Br(),
  
    html.Div([html.H2('Delirium Predictions Explainability',
                      style={'marginLeft': 20, 'color': 'white'})],
             style={
                    'backgroundColor': '#C3CDD6',
                    'padding': '10px 5px',
                    'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px' }),
    dbc.Row([
        dbc.Col([
            dbc.Card(
            dbc.CardBody([


            html.Div("Patient information",
                          style={'font-weight': 'bold', 'font-size': 20}),
            html.Br(),
            dbc.Row([
                dbc.Col(html.Div([
                    html.Label('Enter Patient ID: '),
                    dcc.Input(
                        type="number",
                        debounce=True,
                        value='1',
                        id='pat_id'
                    )
                ]), width={"size": 3}),
                html.Br(),html.Br()],
            style={'padding': '10px 25px'}),
            dbc.Row([
                dash_table.DataTable(
                id = 'table',
                data=[],
                columns=[{'id': 'feature', 'name': 'feature'},{'id': 'value', 'name': 'value'},{'id': 'magnitude', 'name': 'magnitude'},{'type':'numeric'}],
                style_header={'backgroundColor': 'rgb(79, 93, 117)','fontWeight': 'bold'},
                style_cell={
                    'backgroundColor': 'rgb(109, 121, 143)',
                    'color': 'white'
                },
                 style_table={
                'borderRadius': '15px',
                'overflow': 'hidden'
        }
                )
            ],style={'padding': '30px 25px 30px 40px'}),

            dbc.Row([
                    dbc.Button(
                    "Predictive model information",
                    id="collapse-button",
                    className="mb-3",
                    color="primary"
                ),],style={'padding': '20px 25px 30px 40px'}
                )
        ]),style ={'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px',
                    'border': '10px solid rgba(255,255,255,.5)',
                    'background-color': '#F5F8FB',
                    'margin': '10px',
                    'padding': '15px',
                    'position': 'relative'
            })],width = 5),


            # Right hand column containing the summary information for predicted heart disease risk
        dbc.Col([

            dbc.Card(
            dbc.CardBody([
            html.Div("Predicted Post Operative Delirium Risk",
                          style={'font-weight': 'bold', 'font-size': 20}),
            dbc.Row([html.Div(id='main_text', style={'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([html.Div("Factors contributing to predicted likelihood of developing pod",
                              style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row([html.Div(["The figure below indicates the impact (magnitude of increase or decrease in "
                               "probability) of factors on the model prediction of the patient's pod development"],
                              style={'font-size': 16, 'padding': '10px 25px'})]),
            dbc.Row(dcc.Graph(
                id='Metric_2',
                config={'displayModeBar': False}
            ), style={'marginLeft': 10})
         ]),style ={'box-shadow': '2px 2px 2px lightgrey',
                    'border-radius': '5px',
                    'border': '10px solid rgba(255,255,255,.5)',
                    'background-color': '#F5F8FB',
                    'margin': '10px',
                    'padding': '15px',
                    'position': 'relative'
            })], width = 7)
],className="h-75")
]), style = {'background-color': '#FFFFFF'})     
])

@app.callback(
    Output("sunnyburst", "figure"),
    Input("days","value")
    )
def updateSunburst(days):

    df = pd.read_csv('df_sunburst_MF.csv')
    figure = createSunburst(df)
    if days == 10.0:
        figure = createSunburst(df)
    if days == 1.0:
        df = df[df['ASA status'] == 1.0]
        figure = createSunburst(df)
    elif days == 2.0:
        df = df[df['ASA status'] == 2.0]
        figure = createSunburst(df)
    elif days == 3.0:
        df = df[df['ASA status'] == 3.0]
        figure = createSunburst(df)
    elif days == 4.0:
        df = df[df['ASA status'] == 4.0]
        figure = createSunburst(df)
    elif days == 5.0:
        df = df[df['ASA status'] == 5.0]
        figure = createSunburst(df)
    elif days == 6.0:
        df = df[df['ASA status'] == 6.0]
        figure = createSunburst(df)

    return figure
    
       
#enter patient id
@app.callback(
    [Output('Metric_2', 'figure'),
    Output("table", "data")],
    [Input('pat_id','value')]
    )
def createShapEx(patientId):
    data = Data()
    df, f_list = data.get_data()
    X = df.iloc[:,1:]
    y = df.iloc[:,0:1]
    
    X_trainUC, X_testUC, y_trainUC, y_testUC = train_test_split(X, y, test_size=0.2, random_state=19)
    rfc = RandomForestClassifier(max_depth=41, n_estimators=644,
                                random_state=19)
    rfc.fit(X_trainUC,y_trainUC)
    x_new = X_testUC[X_testUC.index == patientId]
    explainer_patient = shap.TreeExplainer(rfc)
    shap_values_patient = explainer_patient.shap_values(x_new)

    updated_fnames = x_new.T.reset_index()
    updated_fnames.columns = ['feature', 'value']
    updated_fnames['magnitude'] = pd.Series(shap_values_patient[1].ravel())
    updated_fnames['shap_abs'] = updated_fnames['magnitude'].abs()
    updated_fnames['shap_importance (%)'] = updated_fnames['magnitude'].apply(lambda x: 100*x/np.sum(updated_fnames['magnitude']))
    updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)
    updated_fnames2 = updated_fnames.sort_values(by=['shap_abs'], ascending=False)
    nf= updated_fnames2.iloc[0:9,0:3]
    nf.reset_index(drop=True, inplace=True)

    show_features = 9
    num_other_features = updated_fnames.shape[0] - show_features
    col_other_name = f"{num_other_features} other features"
    f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
    f_group['feature'] = col_other_name
    plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])
    
    

    plot_range = plot_data['magnitude'].cumsum().max() - plot_data['magnitude'].cumsum().min()
    plot_data['text_pos'] = np.where(plot_data['magnitude'].abs() > (1/9)*plot_range, "inside", "outside")
    plot_data['text_col'] = "white"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['magnitude'] < 0), 'text_col'] = "#3283FE"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['magnitude'] > 0), 'text_col'] = "#F6222E"

    fig2 = go.Figure(go.Waterfall(
    name="",
    orientation="h",
    measure=['absolute'] + ['relative']*show_features,
    base=explainer_patient.expected_value[1],
    textposition=plot_data['text_pos'],
    text=plot_data['magnitude'],
    textfont={"color": plot_data['text_col']},
    texttemplate='%{text:+.2f}',
    y=plot_data['feature'],
    x=plot_data['magnitude'],
    connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
    decreasing={"marker": {"color": "#3283FE"}},
    increasing={"marker": {"color": "#F6222E"}},
    hoverinfo="skip"
    ))
    fig2.update_layout(
        waterfallgap=0.2,
        autosize=False,
        width=800,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5
        ),
        margin={'t': 25, 'b': 50},
        shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1.02,
                xref='x', x0=plot_data['magnitude'].sum()+explainer_patient.expected_value[1],
                x1=plot_data['magnitude'].sum()+explainer_patient.expected_value[1],
                layer="below",
                line=dict(
                    color="black",
                    width=1,
                    dash="dot")
            )
        ]
    )
    fig2.update_yaxes(automargin=True)
    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=explainer_patient.expected_value[1],
        y=-0.12,
        text="Base Value = {:.2f}".format(explainer_patient.expected_value[1]),
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=plot_data['magnitude'].sum()+explainer_patient.expected_value[1],
        y=1.075,
        text="End Value = {:.2f}".format(plot_data['magnitude'].sum()+explainer_patient.expected_value[1]),
        showarrow=False,
        font=dict(color="black", size=14)
    )

    return fig2,nf.to_dict('records')

# Python functions for about navitem-popover
@app.callback(output=Output("about","is_open"), inputs=[Input("about-popover","n_clicks")], state=[State("about","is_open")])
def about_popover(n, is_open):
    if n:
        return not is_open
    return is_open
@app.callback(output=Output("about-popover","active"), inputs=[Input("about-popover","n_clicks")], state=[State("about-popover","active")])
def about_active(n, active):
    if n:
        return not active
    return active


@app.callback(
    Output("graph", "figure"), 
    [Input('model-name', "value")])
def show_plot(name):
    fig_out = 'AUC by gender'
    df = pd.read_csv('df_clean.csv')
    if name == 'Distribution age/gender':
        fig = px.histogram(df, x="age", y="pod", color="sex (0 male, 1 female)",
        marginal="box", # or violin, rug
        hover_data=df.columns)
        fig_out = fig
    elif name == 'AUC by gender':
                # Fit the model

        df_transUCF = df.loc[df['sex (0 male, 1 female)'] == 1.0] 
        df_transUCM = df.loc[df['sex (0 male, 1 female)'] == 0.0] 
        #split for female
        XF = df_transUCF.iloc[:,1:]
        yF = df_transUCF.iloc[:,0:1]

        X_trainUCF, X_testUCF, y_trainUCF, y_testUCF = train_test_split(XF, yF, test_size=0.2, random_state=19)
        
        #split for male
        XF = df_transUCM.iloc[:,1:]
        yF = df_transUCM.iloc[:,0:1]

        X_trainUCM, X_testUCM, y_trainUCM, y_testUCM = train_test_split(XF, yF, test_size=0.2, random_state=19)



        best_modelF = Pipeline(steps=[('preprocessor',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('num',
                                                  Pipeline(steps=[('scaler',
                                                                   StandardScaler())]),
                                                  ['age', 'weight (kg)',
                                                   'height (m)', 'BMI_value'])])),
                #('feat_select', SelectKBest(k=27)),
                ('clf',
                 RandomForestClassifier(max_depth=41, n_estimators=644,
                                        random_state=19))])

        best_modelM = Pipeline(steps=[('preprocessor',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('num',
                                                  Pipeline(steps=[('scaler',
                                                                   StandardScaler())]),
                                                  ['age', 'weight (kg)',
                                                   'height (m)', 'BMI_value'])])),
                #('feat_select', SelectKBest(k=27)),
                ('clf',
                 RandomForestClassifier(max_depth=41, n_estimators=644,
                                        random_state=19))])


        best_modelF.fit(X_trainUCF, y_trainUCF)
        best_modelM.fit(X_trainUCM, y_trainUCM)
        y_scoresF = best_modelF.predict_proba(X_testUCF)[:,1]
        y_scoresM = best_modelM.predict_proba(X_testUCM)[:,1]


        # Create an empty figure, and iteratively add new lines
        # every time we compute a new class
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )


        fprF, tprF, _ = roc_curve(y_testUCF, y_scoresF)
        fprM, tprM, _ = roc_curve(y_testUCM, y_scoresM)
        auc_scoreF = roc_auc_score(y_testUCF, y_scoresF)
        auc_scoreM = roc_auc_score(y_testUCM, y_scoresM)

        nameF = f"female (AUC={auc_scoreF:.2f})"
        nameM = f"male (AUC={auc_scoreM:.2f})"


        fig.add_trace(go.Scatter(x=fprF, y=tprF, name=nameF, mode='lines',fill='tozeroy'))
        fig.add_trace(go.Scatter(x=fprM, y=tprM, name=nameM, mode='lines', fill='tonexty'))

        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            #yaxis=dict(scaleanchor="x", scaleratio=1),
            #xaxis=dict(constrain='domain'),
            margin=dict(
            b=40, #bottom margin 40px
            l=40, #left margin 40px
            r=20, #right margin 20px
            t=20, #top margin 20px
            ),
            height = 550
        )
        fig_out = fig

    return fig_out





