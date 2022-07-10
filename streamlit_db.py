import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from datetime import datetime, date
import plotly.graph_objects as go
#customer imports 
from datautils import load_geo_data
from plotting import  plot_ping_responses, plot_R_chart, plot_top10_on_map
from main import get_ping_stats, top10byMetric
from models import check_seasonality, get_stationarity, FirstOrderDiff, arimamodel, get_arima_orders, perform_GARCH,perform_arima_garch


#from streamlit_option_menu import option_menu


# Page settings
st.set_page_config(page_title='Commnet 2',
                   page_icon = ':bar_chart:', layout ='wide')
st.title(':bar_chart:Project Team 21')
st.markdown('##')



@st.cache(suppress_st_warning=True)
def load_data():
    file1, file2, file3 = ('ping1.csv', 'ping2.csv','ping3.csv')
    df = pd.concat(map(pd.read_csv,[file1, file2, file3]), ignore_index=True)
    return df 

df = load_data()

geo_file_name = 'site_locations.csv'
df_geo = load_geo_data(geo_file_name)

site_selections = df['name'].unique().tolist()
site_selector = st.sidebar.selectbox('Select a site: ', site_selections)
df_selection = df.query('name==@site_selector')


st.markdown(site_selector + "'s Summary:")
site_summary = get_ping_stats(df_selection)
st.write('Packet Loss:'+ str(site_summary['Packet Loss'])+' %.   ',
        'EFsmall Mean:' + str(site_summary['EFsmall Mean'])+' ms.  ',
        'BEsmall Mean:' + str(site_summary['BEsmall Mean'])+' ms.  ',
        'EFlarge Mean:' + str(site_summary['EFlarge Mean'])+' ms.  ',
        'BElarge Mean:' + str(site_summary['BElarge Mean'])+' ms.  ',
        'EFsmall Std Dev:' + str(site_summary['EFsmall Std Dev'])+' ms.  ',
        'BEsmall Std Dev:' + str(site_summary['BEsmall Std Dev'])+' ms.  ',
        'EFlarge Std Dev:' + str(site_summary['EFlarge Std Dev'])+' ms.  ',
        'BElarge Std Dev:' + str(site_summary['BElarge Std Dev'])+' ms.  ',
)

#Plot the site's data
fig1 = plot_ping_responses(df_selection)
st.write(fig1)

payload_selections = ['EFsmall','BEsmall','EFlarge','BElarge']
payload_selector = st.sidebar.selectbox('Select a payload type: ', payload_selections)
df_selection2 = df_selection[[payload_selector]]
fig2 = px.line(df_selection, x='datetime', y=payload_selector, title ='TBD')
fig2.update_layout(
    autosize=False,
    width=1200,
    height=450,
)

st.write(fig2)


metric_selection = ['Packet Loss', 'EFsmall Std Dev','BEsmall Std Dev','EFlarge Std Dev','BElarge Std Dev']

col1, col2 = st.columns(2)

# Top 10 by metric
with col1:
    metric_selector = st.selectbox('Select a metric to view top 10: ', metric_selection)
    top10byslectedMetric = pd.DataFrame(top10byMetric(df, metric_selector),
                          columns=['Site', metric_selector])
    st.write(top10byslectedMetric)

#Plot on map
with col2:
    fig3 = plot_top10_on_map(top10byslectedMetric, df_geo)
    st.write(fig3)


# Range Control Chart
st.header('Range Control Chart')
st.write(site_selector + ' ' + payload_selector)
fig4 = plot_R_chart(df_selection2)
st.write(fig4)


# Run arima and garch
df_selection2[payload_selector] = df_selection2[df_selection2[payload_selector]>0]
df_selection2.dropna(inplace=True)
st.pyplot(get_stationarity(df_selection2))
st.pyplot(FirstOrderDiff(df_selection2))
params = get_arima_orders(df_selection2)
if params[0]==0 and params[1]==0:
    print('Errors in time series data is uncorrelated which means the errors are random')
else:
    st.pyplot(perform_GARCH(df_selection2))








