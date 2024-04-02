import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Stands","result","Players","ELO rank","2024","Qualitfication",'Predict matches - ML'])

with tab1:
    st.header("History of EURO 1960 - present")
    all = pd.read_csv('/Users/admin/Downloads/data/all_Expand.csv', sep =';',index_col=0)
    col1, col2, col3 = st.columns(3)
    col1.metric('Goals',all['GF'].sum())
    col2.metric('Matches',int(all['Matches'].sum()/2))
    col3.metric('Avg goals/match',round(all['GF'].sum()/int(all['Matches'].sum()/2),2))

    all['Points per tournament'].fillna(0, inplace=True)
    all['Points per tournament'] = all['Points per tournament'].round(2).apply(lambda x: '{:.2f}'.format(x))
    st.dataframe(all)

    teams = all.sort_values("Team")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=teams['Team'],
                y=teams['GF'],
                name='Goals for',
                marker_color='rgb(56, 163, 42)'
                ))
    fig.add_trace(go.Bar(x=teams['Team'],
                y=teams['GA'],
                name='Goals against',
                marker_color='rgb(235, 60, 39)'
                ))

    fig.update_layout(
        title='Goals for and against',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Goals',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.2,
        bargroupgap=0.1
    )
    st.plotly_chart(fig)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=teams['Team'],
                         y=teams['Points per tournament'],
                         mode='markers',
                         name='Points per tournament',
                         marker_color='rgb(186, 12, 47)'
                         ))
    fig1.add_trace(go.Bar(x=teams['Team'],
                    y=teams['Participations'],
                    name='Participations',
                    marker_color='rgb(26, 118, 255)'
                ))

    fig1.update_layout(
        title='Participations in EURO and points/tournament',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Participations / Points',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    st.plotly_chart(fig1)

    fig2 = go.Figure()
    fig2=px.bar(teams, x="Team", y=["W","D","L"])
    fig2.update_layout(
        title='Wins, draws and losses per team',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Matches',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    st.plotly_chart(fig2)

with tab2:
    st.header("Players in Euro")
    players = pd.read_csv('/Users/admin/Downloads/data/players_modified.csv', sep =',',index_col=0)
    col1, col2 = st.columns(2)
    active = int(players['Active'].sum())
    most = players.groupby(by='Team').count().sort_values('Player', ascending=False).index[0]
    player = players.groupby(by='Team').count().sort_values('Player', ascending=False)['Player'][0]
    col1.metric('Players',players['Player'].count(),active,delta_color="off", help=f'{active} still playing')
    col2.metric('Most apperences',f'{most} with {player} players')

    st.dataframe(players,  use_container_width=True)

