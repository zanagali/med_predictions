import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import sys
sys.path.insert(1, 'C:/Users/Zana/anaconda3/envs/PIPRA/Work_pipra/app/')
from settings import config, about

from python.data import Data
from python.model import Model

class Result():

    def plot_balance(self):
        new_df = self.dtf['pod']
        pred_df = self.dtf['pred']
        zero_t = new_df[new_df == 0.0].value_counts()
        one_t = new_df[new_df == 1.0].value_counts()
        zero_pred = pred_df[pred_df == 0.0].value_counts()
        one_pred = pred_df[new_df == 1.0].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=['pod'],
            x=[zero_t],
            name='zeros',
            orientation='h',
            marker=dict(
                color='rgba(246, 78, 139, 0.6)',
                line=dict(color='rgba(246, 78, 139, 1.0)', width=3)
            )
        ))
        fig.add_trace(go.Bar(
            y=['pod'],
            x=[one_t],
            name='ones',
            orientation='h',
            marker=dict(
                color='rgba(58, 71, 80, 0.6)',
                line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
            )
        ))
        fig.update_layout(barmode='stack')
        return fig
        

    def probaOverview(self,data):
        model = Model()
        y_score,y_test = model.predictProba(data)
        fig_hist = px.histogram(
            x=y_score, color=y_test, nbins=80,
            labels=dict(color='True Labels', x='Score')
        )
        return fig_hist