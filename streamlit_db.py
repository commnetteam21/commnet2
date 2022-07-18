import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from datetime import datetime, date
import plotly.graph_objects as go
#customer imports 
from datautils import load_geo_data
from plotting import  plot_ping_responses, plot_R_chart,plot_top10_on_map
from main import get_ping_stats, top10byMetric
from models import check_seasonality, get_stationarity, FirstOrderDiff, arimamodel, get_arima_orders, perform_GARCH, MA, acf_plot, pacf_plot


#from streamlit_option_menu import option_menu


# Page settings
st.set_page_config(page_title='Commnet 2',
                   page_icon = ':bar_chart:', layout ='wide')
st.title(':bar_chart:Commnet Ping Responses 2 Team 21 - Spring 2022')
st.markdown('##')

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(suppress_st_warning=True)
def load_data():
    file1, file2, file3 = ('ping1.csv', 'ping2.csv','ping3.csv')
    df = pd.concat(map(pd.read_csv,[file1, file2, file3]), ignore_index=True)
    return df 

df = load_data()

geo_file_name = 'site_locations.csv'
df_geo = load_geo_data(geo_file_name)

site_selections = df['name'].unique().tolist()
site_selections.sort()
site_selector = st.sidebar.selectbox('Select a site: ', site_selections)
df_selection = df.query('name==@site_selector')

st.header('Ping Responses Plot')

st.markdown(site_selector + "'s Summary:")
site_summary = get_ping_stats(df_selection)
st.write('Packet Loss:'+ str(site_summary['Packet Loss'])+' %.       ',
        'EFsmall Mean:' + str(site_summary['EFsmall Mean'])+' ms.    ',
        'BEsmall Mean:' + str(site_summary['BEsmall Mean'])+' ms.    ',
        'EFlarge Mean:' + str(site_summary['EFlarge Mean'])+' ms.    ',
        'BElarge Mean:' + str(site_summary['BElarge Mean'])+' ms.    ',
        'EFsmall Std Dev:' + str(site_summary['EFsmall Std Dev'])+' ms.    ',
        'BEsmall Std Dev:' + str(site_summary['BEsmall Std Dev'])+' ms.    ',
        'EFlarge Std Dev:' + str(site_summary['EFlarge Std Dev'])+' ms.    ',
        'BElarge Std Dev:' + str(site_summary['BElarge Std Dev'])+' ms.    ',
)

#Plot the site's data
fig1 = plot_ping_responses(df_selection)
st.write(fig1)

payload_selections = ['EFsmall','BEsmall','EFlarge','BElarge']
payload_selector = st.sidebar.selectbox('Select a payload type: ', payload_selections)
df_selection2 = df_selection[[payload_selector]]
fig2 = px.line(df_selection, x='datetime', y=payload_selector, title = payload_selector +' Plot')
fig2.update_layout(
    autosize=False,
    width=1500,
    height=450,
)

st.write(fig2)
st.header('Top 10 by selected metric')
metric_selection = ['Packet Loss',
                        'EFsmall Std Dev',
                        'BEsmall Std Dev',
                        'EFlarge Std Dev',
                        'BElarge Std Dev',
                        'EFsmall Mean',
                        'BEsmall Mean',
                        'EFlarge Mean',
                        'BElarge Mean'
                    ]
# if st.button('Click here to view top 10'):
#     col1, col2 = st.columns(2)

#     # Top 10 by metric
#     with col1:
#         metric_selector = st.selectbox('Please select a metric to view top 10: ', metric_selection)
#         st.write(top10byMetric(df, metric_selector))
#         top10byslectedMetric = pd.DataFrame(
#                                top10byMetric(df, metric_selector),
#                                columns=['Site', metric_selector])
#         st.write(top10byslectedMetric)

#     #Plot on map
#     with col2:
#         fig3 = plot_top10_on_map(top10byslectedMetric, df_geo)
#         st.write(fig3)


metric_selector = st.sidebar.selectbox('Please select a metric to view top 10: ', metric_selection)

if st.button('Click here to view top 10'):
    metric_ranking = top10byMetric(df, metric_selector)
    top10byselectedMetric = pd.DataFrame(
                                metric_ranking,
                                columns=['Site', metric_selector])
    col1, col2 = st.columns(2)
     
    with col1:
        st.markdown('Top 10 by '+metric_selector)
        st.write(top10byselectedMetric.head(10))
    with col2:
        st.markdown('Bottom 10 by '+metric_selector)
        st.write(top10byselectedMetric.tail(10))                            
    fig4 = plot_top10_on_map(top10byselectedMetric[:20],df_geo)
    st.write(fig4) 

# Range Control Chart
st.header('Range Control Chart')
st.write(site_selector + ' ' + payload_selector)
fig5 = plot_R_chart(df_selection2)
st.write(fig5)



# Run arima and garch
st.header('Statistical Modeling using ARIMA and GARCH')

if st.button('Click here to run Arima and GARCH'):
    df_selection2[payload_selector] = df_selection2[df_selection2[payload_selector]>0]
    df_selection2.dropna(inplace=True)
    st.markdown('Seasonality Investigation')
    st.pyplot(check_seasonality(df_selection2))
    st.markdown('Stationarity Investigation')
    stationary_fig, result = get_stationarity(df_selection2)
    st.pyplot(stationary_fig)
    st.write('ADF Statistic: {}'.format(result[0]))
    st.write('p-value: {}'.format(result[1]))
    st.write('Critical Values:')
    for key, value in result[4].items():
        st.write('\t{}: {}'.format(key, value))

    st.pyplot(FirstOrderDiff(df_selection2))
    col1, col2 = st.columns(2)
    with col1:
        acf_plot(df_selection2)
    with col2:
        pacf_plot(df_selection2)
    params = get_arima_orders(df_selection2)
    st.markdown('Result:')
    st.write('Optimal Order is: (p,d,q) = ', params)
    st.pyplot(perform_GARCH(df_selection2,params))











