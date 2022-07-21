import pandas as pd 
import numpy as np
import plotly.graph_objects as go




def load_geo_data(file_name):
    df = pd.read_csv(file_name)
    return df



def get_ping_stats(df)->dict:
    EFsmall_loss = len(df[df['EFsmall']==0])
    BEsmall_loss = len(df[df['BEsmall']==0])
    EFlarge_loss = len(df[df['EFlarge']==0])
    BElarge_loss = len(df[df['BElarge']==0])
    Packet_loss = round(100*(EFsmall_loss+BEsmall_loss+EFlarge_loss+BElarge_loss)/len(df),2)

    EFsmall_mean = round(df['EFsmall'].mean(),2)
    BEsmall_mean = round(df['BEsmall'].mean(),2)
    EFlarge_mean = round(df['EFlarge'].mean(),2)
    BElarge_mean = round(df['BElarge'].mean(),2)

    EFsmall_std = round(df['EFsmall'].std(),2)
    BEsmall_std = round(df['BEsmall'].std(),2)
    EFlarge_std = round(df['EFlarge'].std(),2)
    BElarge_std = round(df['BElarge'].std(),2)

    summary_dict = {'Packet Loss': Packet_loss,
                    'EFsmall Mean': EFsmall_mean,
                    'BEsmall Mean': BEsmall_mean,
                    'EFlarge Mean': EFlarge_mean,
                    'BElarge Mean': BElarge_mean,
                    'EFsmall Std Dev': EFsmall_std,
                    'BEsmall Std Dev': BEsmall_std,
                    'EFlarge Std Dev': EFlarge_std,
                    'BElarge Std Dev': BElarge_std,
    }

    return summary_dict

def top10byMetric(df, chosen_metric:str)->list: #df is collective data
    site_names = df['name'].unique().tolist()
    top_10_list = []
    for idx, site in enumerate(site_names):
        site_df = df[df['name']==site]
        metric = get_ping_stats(site_df)
        if metric['Packet Loss'] < 350:
            top_10_list.append((site,metric[chosen_metric]))
    top_10_list.sort(key=lambda x:x[1], reverse=True) # change back to True
    return top_10_list



