import pandas as pd 
import numpy as np
import plotly.graph_objects as go


####
def plot_ping_responses(df):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=df.datetime,
            y=df.EFsmall, 
            name='EFsmall'
    ))

    fig.add_trace(go.Scatter(
            x=df.datetime,
            y=df.BEsmall,
            name='BEsmall'
    ))

    fig.add_trace(go.Scatter(
            x=df.datetime,
            y=df.EFlarge,
            name='EFlarge'
    ))

    fig.add_trace(go.Scatter(
            x=df.datetime,
            y=df.BElarge,
            name='BElarge'
    ))

    fig.update_layout(
    autosize=False,
    width=1500,
    height=450,
    )

    return fig


def plot_R_chart(df):
    # df being df after site has been sited and payload has been chosen
    # original size of df is 2873. We need to increase to 2875 to split the data into 115 groups of 25
    df = df.append({df.columns[0]:float(df[[df.columns[0]]] \
                      .iloc[2870:2875].mean())}, ignore_index=True
    )
    df = df.append({df.columns[0]:float(df[[df.columns[0]]] \
                      .iloc[2870:2875].mean())}, ignore_index=True
    )
    
    # 2875/25 = 115
    groupNum = 115
    groupSize = 25
    df_arr = df.to_numpy().reshape(groupNum,groupSize)
    #https://web.mit.edu/2.810/www/files/readings/ControlChartConstantsAndFormulae.pdf
    D3_25 = 0.459 #Constant for LCL of subgroup size = 25
    D4_25 = 1.541 #Constant for UCL of subgroup size = 25

    range_list = [(df_arr[i].max()-df_arr[i].min()) for i in range(115)]

    R_bar = np.mean(range_list)
    LCL = R_bar*D3_25
    UCL = R_bar*D4_25
    R_bar_line = [R_bar for i in range(len(range_list))]
    LCL_line = [LCL for i in range(len(range_list))]
    UCL_line = [UCL for i in range(len(range_list))]

    #Plot all the lines
    fig = go.Figure()
    fig.add_trace(go.Scatter(
            x=list(range(len(range_list))),
            y=range_list,
            mode='lines+markers',
            name='Range Values'  
    ))

    fig.add_trace(go.Scatter(
            x=list(range(len(range_list))),
            y=R_bar_line,
            mode='markers',
            name='Range Mean'
    ))

    fig.add_trace(go.Scatter(
            x=list(range(len(range_list))),
            y = LCL_line,
            mode='lines',
            name='LCL',                       
    ))

    fig.add_trace(go.Scatter(
            x=list(range(len(range_list))),
            y = UCL_line,
            mode='lines',
            name='UCL'
    ))

    fig.update_layout(
            autosize=False,
            width=1500,
            height=450,
    )
    return fig


def plot_top10_on_map(top10df, geo_df):
    top10df.Site = top10df.Site.str.replace('[^a-zA-Z -]+$', '')
    top10geo = pd.merge(
                    left=top10df, 
                    right=geo_df,
                    how = 'left',
                    left_on ='Site',
                    right_on = 'report_site_name'
                    ) \
                    [['Site','latitude','longitude']]\
                    .dropna()
                      
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lon=top10geo['longitude'],
        lat=top10geo['latitude']
    ))
    fig.update_layout(
        title_text='top 10 by selection',
        showlegend=True,
        geo = dict(
            scope='usa',
            landcolor = 'rgb(217,217,217)',
        )
    )
    return fig