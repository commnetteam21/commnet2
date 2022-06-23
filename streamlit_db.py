import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
from datetime import datetime, date
import plotly.graph_objects as go 
from main import get_ping_stats
#from streamlit_option_menu import option_menu



st.set_page_config(page_title='Commnet 2',
                   page_icon = ':bar_chart:', layout ='wide')
st.title(':bar_chart:Project Team 21')
st.markdown('##')

@st.cache(suppress_st_warning=True)

def load_data():
    file_path1 = 'ping1.csv'
    file_path2 = 'ping2.csv'
    file_path3 = 'ping3.csv'
    df = pd.concat(map(pd.read_csv,[file_path1, file_path2, file_path3]), ignore_index=True)
    return df 

df = load_data()



site_selections = df['name'].unique().tolist()
payload_selections = ['EFsmall','BEsmall','EFlarge','BElarge']

site_selector = st.selectbox('Select a site: ', site_selections)


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
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_selection['datetime'], y=df_selection['EFsmall'], name='EFsmall'))
fig1.add_trace(go.Scatter(x=df_selection['datetime'], y=df_selection['BEsmall'], name='BEsmall'))
fig1.add_trace(go.Scatter(x=df_selection['datetime'], y=df_selection['EFlarge'], name = 'EFlarge'))
fig1.add_trace(go.Scatter(x=df_selection['datetime'], y=df_selection['BElarge'], name = 'BElarge'))
fig1.update_layout(
    autosize=False,
    width=1500,
    height=600,
)
st.write(fig1)



payload_selector = st.selectbox('Select a payload type: ', payload_selections)

fig2 = px.line(df_selection, x='datetime', y=payload_selector, title ='TBD')
fig2.update_layout(
    autosize=False,
    width=1500,
    height=600,
)

st.write(fig2)







