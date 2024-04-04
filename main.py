import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def fig1():

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(x=teams.index.get_level_values("Country"),
                y=teams['GF'],
                name='Goals for',
                marker_color='green'
                ))
    fig1.add_trace(go.Bar(x=teams.index.get_level_values("Country"),
                y=teams['GA'],
                name='Goals against',
                marker_color='red'
                ))
    fig1.update_layout(
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
    return st.plotly_chart(fig1)

def fig2():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=teams.index.get_level_values("Country"),
                              y=teams['Points per tournament'],
                              mode='markers',
                              name='Points per tournament',
                              marker_color='red'
                              ))
    fig2.add_trace(go.Bar(x=teams.index.get_level_values("Country"),
                          y=teams['Participations'],
                          name='Participations',
                          marker_color='blue'
                          ))

    fig2.update_layout(
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
    return st.plotly_chart(fig2)

def fig3():
    fig3 = go.Figure()
    fig3 = px.bar(teams, x=teams.index.get_level_values("Country"), y=["W", "D", "L"],
                  labels={"variable": "Result"})
    fig3.update_layout(
        title='Wins, draws and losses per team',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Matches',
            titlefont_size=16,
            tickfont_size=14,
        ),
        xaxis=dict(
            title='',
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
    return st.plotly_chart(fig3)

st.header("Dashboard - EURO")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Stands","Players","Result","ELO rank","2024","Qualitfication",'Predict matches - ML'])

with tab1:
    st.subheader("🗓️ EURO 1960 - present")
    st.write('')
    all = pd.read_csv('/Users/admin/Downloads/data/all.csv', sep =';',index_col=[0,1])
    trophy = lambda x: (x * '🏆')
    all['Trophy'] = all['Trophy'].apply(trophy)
    matches = int(all['Matches'].sum()/2)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Goals',all['GF'].sum())
    col2.metric('Matches',matches)
    col3.metric('Avg goals/match',round(all['GF'].sum()/matches,2))
    col4.metric('Teams',all['Matches'].count())
    all['Points per tournament'].fillna(0, inplace=True)
    all['Points per tournament'] = all['Points per tournament'].round(2).apply(lambda x: '{:.2f}'.format(x))
    st.dataframe(
        all,
        column_config={
            "Points": st.column_config.ProgressColumn(
                "Points",
                format="%f",
                min_value=0,
                max_value=100,
            ),
            "Points per tournament": st.column_config.ProgressColumn(
                "Points per tournament",
                help='aaa',
                format="%.2f",
                min_value=0,
                max_value=10,
            ),
        }
    )

    teams = all.sort_index(level='Country')

    fig1()
    fig2()
    fig3()

with tab2:
    st.header("Players in Euro")
    players = pd.read_csv('/Users/admin/Downloads/data/players_modified.csv', sep =',',index_col=0)
    players['Active'].fillna(False, inplace=True)
    print(players['Active'])
    col1, col2 = st.columns(2)
    active = int(players['Active'].sum())
    most = players.groupby(by='Team').count().sort_values('Player', ascending=False).index[0]
    player = players.groupby(by='Team').count().sort_values('Player', ascending=False)['Player'][0]
    spain = players.loc[players['Team']==most,'Matches'].sum()
    on = st.toggle('Active player')
    if on:
        col1.metric('Players',players['Player'].count(),active,delta_color="off", help=f'{active} still playing')
    else:
        col1.metric('Players', players['Player'].count())
    col2.metric('Most apperences',f'{spain} with {player} players {spain}')
    players = players.reset_index(drop=True)
    players.index += 1
    if on:
        st.dataframe(players.loc[players['Active']==True],  use_container_width=True)
    else:
        st.dataframe(players,  use_container_width=True)

    goals = pd.read_csv('/Users/admin/Downloads/data/goals_modified.csv', sep=',', index_col='Index')
    goals['AVG'] = goals['AVG'].round(2).apply(lambda x: '{:.2f}'.format(x))

    col3, col4 = st.columns(2)
    most_goals_nation = goals.groupby(by='Team')[['Team','Goals']].sum().sort_values(by='Goals',ascending=False).index[0]
    most_goals = goals.groupby(by='Team')[['Team', 'Goals']].sum().sort_values(by='Goals', ascending=False)['Goals'][0]
    col3.metric('Players', goals['Player'].count())
    col4.metric('Most goals',f'{most_goals_nation} with {most_goals} goals')
    goals['AVG']=goals['AVG'].astype(float)
    values = st.slider(
        'Select a range of avg goals per match',
        0.0, 2.5, (0.0, 2.5))
    on1 = st.toggle('Active player1')
    if on1:
        st.dataframe(goals.loc[((goals['AVG'].between(float(values[0]), float(values[1]))) & (goals['Active']==True))], use_container_width=True)
    else:
        st.dataframe(goals.loc[goals['AVG'].between(float(values[0]),float(values[1]))], use_container_width=True)

    fig4 = go.Figure()

    # Określenie kolorów na podstawie wartości 'Active'
    colors = [1 if active else 0 for active in players['Active']]

    fig4.add_trace(go.Scatter(
        x=goals['Matches'],
        y=goals['Goals'],
        mode='markers',
        hoverinfo='text',
        text=goals['Player'],
        name='all',
        marker=dict(
            color=colors,
        )
    ))

    # Aktualizacja ustawień wykresu
    fig4.update_layout(
        title='Goals and matches',
        xaxis=dict(title='Matches', titlefont_size=16, tickfont_size=14),
        yaxis=dict(title='Goals', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )


    # Wyświetlanie wykresu w Streamlit
    st.plotly_chart(fig4)

with tab3:
    result = pd.read_csv('/Users/admin/Downloads/data/result_modified.csv',sep=',',index_col=0)
    tab31, tab32 = st.columns(2)
    tab31.metric('Matches',result['Extra Time'].sum(),'was ended by Extra Time',delta_color="off")
    tab32.metric('Penatly',result['Penatly'].sum(),round(result['Penatly'].sum()/result['Euro'].count(),2))
    result['Euro']=result['Euro'].astype(str)
    filtering = st.radio(
        "How match was enderd",
        ["regular time", "extra time", "penatly"],
        key="visibility"
    )
    teams=sorted(result['Euro'].unique())
    if filtering == 'regular time':
        euro = st.selectbox('Select EURO:',teams)


    if filtering == 'extra time':
        st.dataframe(result.loc[result['Extra Time']==True])
    elif filtering == 'penatly':
        st.dataframe(result.loc[result['Penatly'] == True])
    else:
        st.dataframe(result.loc[result['Euro']==euro])

    result['H'] = result['H'].astype(int)
    result['A'] = result['A'].astype(int)
    result['SUM'] = result['H'] + result['A']
    euros = result.groupby('Euro').agg({'SUM': 'sum', 'H': 'count'})
    euros['AVG'] = round(euros['SUM'] / euros['H'], 2)
    euros.rename(columns={'H': 'COUNT'})

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=euros.index, y=euros['SUM'], mode='lines+markers', name='Goals', yaxis='y'))
    fig.add_trace(go.Scatter(x=euros.index, y=euros['AVG'], mode='markers', name='AVG', marker=dict(color='green', size=10),yaxis='y2'))

    fig.update_layout(yaxis=dict(title='Goals'), yaxis2=dict(title='AVG', overlaying='y', side='right'))
    fig.update_xaxes(tickmode='array', tickvals=euros.index)
    fig.update_layout(xaxis_title='EURO', title='Goals in EURO and avg goals/EURO')

    st.plotly_chart(fig)

    stages = result.groupby('Stage').agg({'SUM': 'sum', 'H': 'count'})
    stages['AVG'] = round(stages['SUM'] / stages['H'], 2)
    stages.rename(columns={'H': 'COUNT'})
    order = ['Group', 'Round of 16', 'Quarter-finals', 'Semi-finals', '3rd place', 'Final']
    stages = stages.reindex(order)

    fig32 = go.Figure()
    fig32.add_trace(go.Bar(x=stages.index, y=stages['H'], name='Goals', yaxis='y'))
    fig32.add_trace(go.Scatter(x=stages.index, y=stages['AVG'], mode='markers', name='AVG', marker=dict(color='red', size=12),yaxis='y2'))
    fig32.update_layout(yaxis=dict(title='Matches'), yaxis2=dict(title='AVG', overlaying='y', side='right'))
    fig32.update_xaxes(tickmode='array', tickvals=stages.index)
    fig32.update_layout(title='Matches per stage and avg goals/stage')

    st.plotly_chart(fig32)







with tab4:
    elo_start = pd.read_csv('/Users/admin/Downloads/data/ELO_start.csv', sep =';')
    elo_end = pd.read_csv('/Users/admin/Downloads/data/ELO_end.csv', sep =';')
    elo = pd.merge(elo_start, elo_end, on=['Euro', 'Team'], how='inner')
    team = sorted(elo['Team'].unique())
    option = st.selectbox(
        "Which country do you interested in?",
        team,
        index=None,
        placeholder="Select country...",
    )
    elo['Change']=elo['Rating_y']-elo['Rating_x']
    elo['Euro']=elo['Euro'].astype(str)
    order = ['Euro','Team','Rating_x','Local_x','Global_x','Rating_y','Local_y','Global_y','Change']
    elo=elo[order]
    if option:
        st.dataframe(elo.loc[elo['Team']==option])
        col1, col2 = st.columns(2)
        col1.metric('Best Euro:', int(elo.loc[(elo['Change'] == elo['Change'].loc[elo['Team'] == option].max()) & (elo['Team'] == option)]['Euro']))
        col2.metric('Worst Euro:', int(elo.loc[(elo['Change'] == elo['Change'].loc[elo['Team'] == option].min()) & (elo['Team'] == option)]['Euro']))
    else:
        st.dataframe(elo, use_container_width=True)

    chart_data = pd.DataFrame(elo['Euro'],elo['Rating_x'])
    st.line_chart(chart_data)

with tab5:
    tabA, tabB, tabC, tabD, tabE, tabF = st.tabs(["Group A","Group B","Group C","Group D","Group E","Group F"])
    team_A = ['Germany', 'Scotland', 'Switzerland', 'Hungary']
    team_B = ['Spain', 'Italy', 'Croatia', 'Albania']
    team_C = ['Denmark', 'England', 'Slovenia', 'Serbia']
    team_D = ['France', 'Poland', 'Netherlands', 'Austria']
    team_E = ['Ukraine', 'Romania', 'Slovakia', 'Belgium']
    team_F = ['Portugal', 'Turkey', 'Czech Republic', 'Georgia']
    all = [element for sublist in [team_A,team_B,team_C,team_D,team_E,team_F] for element in sublist]
    all_df = pd.DataFrame(all, columns=['Country'])
    all_df['Euro 2024']=True


    with tabA:
        A = pd.read_csv('/Users/admin/Downloads/data/h2h/groupA_H2H.csv', sep=';', names=['Date','Match','Result','Score','Type'])
        st.dataframe(A, use_container_width=True)

with tab6:
    st.subheader('Passes for Country')
    passes = pd.read_csv('/Users/admin/Downloads/data/2024Q_passes.csv', sep=';')
    passes = pd.merge(passes, all_df, on=['Country'], how='left')
    passes['Euro 2024'].fillna(False, inplace=True)
    passes = passes.reset_index(drop=True)
    passes.index += 1
    st.dataframe(passes)
    shots = pd.read_csv('/Users/admin/Downloads/data/2024Q_shots.csv', sep=';')
    goals = pd.read_csv('/Users/admin/Downloads/data/2024Q_goals.csv', sep=';')
    goals.drop(columns=['#'], inplace=True, axis=0)
    shots = pd.merge(shots, goals, on=['Country'], how='left')
    shots = pd.merge(shots, all_df, on=['Country'], how='left')
    shots['Euro 2024'].fillna(False, inplace=True)
    shots = shots.reset_index(drop=True)
    shots.index += 1
    st.dataframe(shots)
    fouls = pd.read_csv('/Users/admin/Downloads/data/2024Q_fouls.csv', sep=';')
    fouls.drop(columns=['#'], inplace=True, axis=0)
    yellow_cards = pd.read_csv('/Users/admin/Downloads/data/2024Q_yellow_cards.csv', sep=';')
    yellow_cards.drop(columns=['#'],inplace=True,axis=0)
    red_cards = pd.read_csv('/Users/admin/Downloads/data/2024Q_red_cards.csv', sep=';')
    red_cards.drop(columns=['#'], inplace=True, axis=0)
    cards = pd.merge(fouls, yellow_cards, on=['Country'], how='left')
    cards = pd.merge(cards, red_cards, on=['Country'], how='left')
    cards['Red cards'].fillna(0, inplace=True)
    cards = pd.merge(cards, all_df, on=['Country'], how='left')
    cards['Euro 2024'].fillna(False, inplace=True)
    cards = cards.reset_index(drop=True)
    cards.index += 1
    st.dataframe(cards)
    st.divider()
    fig4 = go.Figure()

    colors = [1 if val else 0 for val in passes['Euro 2024']]

    fig4.add_trace(go.Scatter(
        x=passes['Pass Accuracy (avg)'],
        y=passes['Total passes (avg)'],
        mode='markers',
        hoverinfo='text',
        text=passes['Country'],
        name='all',
        marker=dict(
            color=colors,
            colorscale='Viridis'
        )
    ))

    print(colors)

    fig4.update_layout(
        title='Goals and matches',
        xaxis=dict(title='Pass Accuracy (avg)', titlefont_size=16, tickfont_size=14),
        yaxis=dict(title='Total passes (avg)', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )

    st.plotly_chart(fig4)

with tab7:
    pass
