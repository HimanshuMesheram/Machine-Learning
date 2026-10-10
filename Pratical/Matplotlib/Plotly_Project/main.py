import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('india.csv')

list_of_state = list(df['State'].unique())
list_of_state.insert(0,'Overall India')

st.sidebar.title("India's Data Viz")
selected_state = st.sidebar.selectbox('Select the state',list_of_state)
primary = st.sidebar.selectbox('Select primary parameter',sorted(df.columns[6:]))
secondary = st.sidebar.selectbox('Select secondary parameter',sorted(df.columns[6:]))

plot = st.sidebar.button('Plot Graph')

if plot:
    if selected_state == 'Overall India':
        #plot
        fig = px.scatter_map(df, lat="Latitude", lon="Longitude",size=primary,color=secondary, zoom=3,size_max=35,
                  map_style="carto-positron",width=1200,height=700)
        
        st.plotly_chart(fig)
    else:
        # plot for state
        state_df = df[df['State'] == selected_state]
        fig = px.scatter_map(state_df, lat="Latitude", lon="Longitude",size=primary,color=secondary, zoom=3,size_max=35,
                  map_style="carto-positron",width=1200,height=700, hover_name='District')
        
        st.plotly_chart(fig)
