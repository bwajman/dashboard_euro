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

def fig4(df):
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=sorted(df["Country"].unique()),
                          y=df.groupby(by="Country")["Matches"].sum(),
                          name='Matches',
                          marker_color='blue'
                          ))
    fig4.add_trace(go.Scatter(x=sorted(df["Country"].unique()),
                              y=df.groupby(by="Country")["Player"].count(),
                              mode='markers+text',
                              name='Players',
                              marker_color='red'
                              ))
    fig4.update_layout(
        title='Matches and players per nations',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Matches / Players',
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
    return st.plotly_chart(fig4)

def fig56(df,df1):
    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(x=df.index, y=df['SUM'], mode='lines+markers', name='Goals', yaxis='y'))
    fig5.add_trace(go.Scatter(x=df.index, y=df['AVG'], mode='markers', name='AVG', marker=dict(color='green', size=10),yaxis='y2'))

    fig5.update_layout(yaxis=dict(title='Goals'), yaxis2=dict(title='AVG', overlaying='y', side='right'))
    fig5.update_xaxes(tickmode='array', tickvals=euros.index)
    fig5.update_layout(xaxis_title='EURO', title='Goals in EURO and avg goals/EURO')

    fig6 = go.Figure()

    fig6.add_trace(go.Bar(x=df1.index, y=df1['H'], name='Goals', yaxis='y'))
    fig6.add_trace(go.Scatter(x=df1.index, y=df1['AVG'], mode='markers', name='AVG', marker=dict(color='red', size=12),yaxis='y2'))

    fig6.update_layout(yaxis=dict(title='Matches'), yaxis2=dict(title='AVG', overlaying='y', side='right'))
    fig6.update_xaxes(tickmode='array', tickvals=df1.index)
    fig6.update_layout(title='Matches per stage and avg goals/stage')

    return st.plotly_chart(fig5), st.plotly_chart(fig6)

def fig7(df):
    fig7 = go.Figure()

    colors = ['red' if val < 0 else 'green' for val in df['ELO change']]

    fig7.add_trace(go.Bar(
        x=df['Euro'],
        y=df['ELO change'],
        name='Goals',
        yaxis='y',
        marker=dict(color=colors)  # Set colors according to the condition
    ))

    fig7.update_xaxes(tickmode='array', tickvals=df['Euro'])
    fig7.update_layout(xaxis_title='EURO',yaxis_title='Goals', title='Change ELO rank in EURO')

    return st.plotly_chart(fig7)

def fig8(df):
    fig8 = go.Figure()

    colors = [1 if active else 0 for active in df['Active']]

    fig8.add_trace(go.Scatter(
        x=df['Matches'],
        y=df['Goals'],
        mode='markers',
        hoverinfo='text',
        text=df['Player'] + '<br>' + 'Matches: ' + df['Matches'].astype(str) + '<br>' + 'Goals: ' + df['Goals'].astype(str) + ' (' + df['Penatlies'].astype(str) + ') ' + '<br>' + 'AVG: ' + df['AVG'].astype(str),
        name='all',
        marker=dict(
            color=colors,
            colorscale='Viridis'
        )
    ))

    fig8.update_layout(
        title='Goals and matches',
        xaxis=dict(title='Matches', titlefont_size=16, tickfont_size=14),
        yaxis=dict(title='Goals', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )

    st.plotly_chart(fig8)

def fig9(W,D,L):
    fig9 = go.Figure()

    fig9.add_trace(go.Pie(
        labels=['W','D','L'],
        values= [W,D,L],
        hole=0.3
    ))

    fig9.update_layout(title='Change ELO rank in EURO')

    st.plotly_chart(fig9)


def most_appear(df):
    count = df['Player'].count()
    nation = df.groupby(by='Country').count().sort_values('Player', ascending=False).index[0]
    player = df.groupby(by='Country').count().sort_values('Player', ascending=False)['Player'][0]
    caps = df.loc[df['Country'] == nation, 'Matches'].sum()
    return count, nation, player, caps

def most_goals(df):
    if len(df)>0:
        gplayers = df['Player'].count()
        ggoals = df['Goals'].sum()
        gnation = df.groupby(by='Country')[['Country','Goals']].sum().sort_values(by='Goals',ascending=False).index[0]
        gplayer = df.loc[df['Country'] == gnation,'Player'].count()
        gcaps = df.loc[df['Country'] == gnation, 'Goals'].sum()
        return gplayers, ggoals, gnation, gplayer, gcaps

def view_group(team,link):
        col1, col2 = st.columns(2)
        with col1:
            home = st.selectbox(
                'How would you like to be contacted?',
                team)
        with col2:
            team.remove(home)
            away = st.selectbox(
                'How would you like to be contacted?',
                team)

        A = pd.read_csv(link, sep=',')
        st.write(f'You selected: {home} vs {away}')
        A.drop(A.columns[0],axis=1,inplace=True)
        A = A.reset_index(drop=True)
        A.index += 1
        A= A.loc[(A['Home'].str.contains(home) & (A['Away'].str.contains(away))) | ((A['Home'].str.contains(away)) & (A['Away'].str.contains(home)))]
        H = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'H')) | ((A['Home'].str.contains(away)) & (A['Result'] == 'A'))].count()
        D = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'D')) | ((A['Home'].str.contains(away)) & (A['Result'] == 'D'))].count()
        L = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'A')) | ((A['Home'].str.contains(away)) & (A['Result'] == 'H'))].count()
        GF = A['H'].loc[A['Home'].str.contains(home)].sum() + A['A'].loc[A['Away'].str.contains(home)].sum()
        GA = A['A'].loc[A['Home'].str.contains(home)].sum() + A['H'].loc[A['Away'].str.contains(home)].sum()
        st.caption(f'{home} has {H} win {D} draws and {L} lost times against {away} scored {GF} and get {GA} goals')
        col3, col4 = st.columns(2)
        with col3:
            fig9 = go.Figure()

            fig9.add_trace(go.Pie(
                labels=['Wins', 'Draws', 'Losses'],
                values=[H, D, L],
                hole=0.15
            ))
            fig9.update_layout(
                width=300,
                height=400
            )
            st.plotly_chart(fig9)

        with col4:
            fig10 = go.Figure()
            fig10.add_trace(go.Pie(
                labels=['Goals scored', 'Goals against'],
                values=[GF, GA],
                hole=0.15,
            ))

            fig10.update_layout(
                width=300,
                height=400
            )

            st.plotly_chart(fig10)
        st.dataframe(A, use_container_width=True)


st.header("Dashboard - EURO")

team_A = ['Germany', 'Scotland', 'Switzerland', 'Hungary']
team_B = ['Spain', 'Italy', 'Croatia', 'Albania']
team_C = ['Denmark', 'England', 'Slovenia', 'Serbia']
team_D = ['France', 'Poland', 'Netherlands', 'Austria']
team_E = ['Ukraine', 'Romania', 'Slovakia', 'Belgium']
team_F = ['Portugal', 'Turkey', 'Czech Republic', 'Georgia']
all = [element for sublist in [team_A, team_B, team_C, team_D, team_E, team_F] for element in sublist]
all_df = pd.DataFrame(all, columns=['Country'])
all_df['Euro 2024'] = True

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🗓️ stands","⚽️ players","🏅 history result","📈 ELO","🏆 EURO 2024","✅ qualitfication",'🤖 predict matches - ML'])

with tab1:
    st.subheader("EURO 1960 - present")
    st.write('')
    all = pd.read_csv('/Users/admin/Downloads/data/all.csv', sep =';')
    all.loc[all['Country']=='Türkiye','Country']='Turkey'
    all = pd.merge(all, all_df, on=['Country'], how='left')
    all['Euro 2024'].fillna(False, inplace=True)
    last_column = all.pop(all.columns[-1])
    all.insert(2, last_column.name, last_column)
    all.set_index([all.columns[0], all.columns[1]], inplace=True)
    all.index.names = ['No','Country']
    trophy = lambda x: (x * '🏆')
    all['Trophy'] = all['Trophy'].apply(trophy)
    col1, col2, col3, col4 = st.columns(4)
    euro2024 = st.toggle('Euro 2024')
    if euro2024:
        all=all.loc[all['Euro 2024']==True]
        st.caption('* from 2024 which partcipants in EURO - Serbia didnt play at EURO and Gerogia')
    matches = int(all['Matches'].sum()/2)
    col1.metric('Goals',all['GF'].sum())
    col2.metric('Matches',matches)
    col3.metric('Avg goals/match',round(all['GF'].sum()/matches,2))
    col4.metric('Teams',len(all.index.get_level_values(1)))
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

    teams=sorted(players['Country'].unique())

    goals = pd.read_csv('/Users/admin/Downloads/data/goals_modified.csv', sep=',',index_col=1)
    goals['AVG'] = goals['AVG'].round(2).apply(lambda x: '{:.2f}'.format(x))
    goals['AVG'] = goals['AVG'].astype(float)
    goals.drop(goals.columns[0], inplace=True ,axis=1)

    select = st.checkbox('I want to select teams')
    options = st.multiselect(
        'What country do you want to check?',
        teams,
        teams[24],
        disabled=not select)

    on = st.toggle('Active player')
    col1, col2, col3, col4 = st.columns(4)
    count, nation, player, caps = most_appear(players)
    gplayers, ggoals, gnation, gplayer, gcaps = most_goals(goals)

    if on and select and len(options)>0:
        players = players.loc[(players['Active'] == True) & (players['Country'].isin(options))]
        count, nation, player, caps = most_appear(players)
    elif on and not select:
        players = players.loc[players['Active']==True]
        count, nation, player, caps = most_appear(players)
    elif on and select and len(options)==0:
        players = players.loc[players['Active']==True]
        count, nation, player, caps = most_appear(players)
    elif select and len(options)>0:
        players = players.loc[players['Country'].isin(options)]
        count, nation, player, caps = most_appear(players)
    else:
        players = players
        count, nation, player, caps = most_appear(players)

    col1.metric('Players', f'{count}')
    col2.metric('Most popular nation is',f'{nation}')
    col3.metric('with players', f'{player} ')
    col4.metric('and apperances', f'{caps}')

    st.dataframe(players,  use_container_width=True)

    with st.expander("Show data of chart"):
        fig4(players)

    st.divider()

    values = st.slider(
        'Select a range of avg goals per match',
        0.0, 2.5, (0.0, 2.5))

    if on:
        goals = goals.loc[((goals['AVG'].between(float(values[0]), float(values[1]))) & (goals['Active']==True))]
        if len(goals)>0:
            gplayers, ggoals, gnation, gplayer, gcaps = most_goals(goals)

    if len(goals) > 0:
        goals = goals.loc[goals['AVG'].between(float(values[0]), float(values[1]))]
        try:
            gplayers, ggoals, gnation, gplayer, gcaps = most_goals(goals)
            col5, col6, col7, col8 = st.columns(4)
            col5.metric('Players', gplayers)
            col6.metric('Goals', ggoals)
            col7.metric('Most popular nation is', gnation)
            col8.metric('with players', gplayer)
            col8.metric('and goals', gcaps)
            goals['AVG'] = goals['AVG'].round(2).apply(lambda x: '{:.2f}'.format(x))
            goals['AVG']= goals['AVG'].astype(str)
            st.dataframe(goals, use_container_width=True)
            fig8(goals)
        except:
            st.error("No records to display :(")






with tab3:
    result = pd.read_csv('/Users/admin/Downloads/data/result_modified.csv',sep=',',index_col=0)
    col1, col2 = st.columns(2)
    col1.metric('Matches was ended by extra time',result['Extra Time'].sum())
    col2.metric('Matches was ended by penatlies',result['Penatly'].sum())
    result['Euro']=result['Euro'].astype(str)

    euro = sorted(result['Euro'].unique())

    with col2:
        check_euro = st.checkbox('Want to check Euro')

    with col1:
        filtering = st.radio(
        "Select matches:",
        ["no specify", "by extra time", "by penatlies"],
        key="visibility",
        disabled= check_euro
        )
    with col2:
        option = st.selectbox(
        "Which euro do you want to check?",
        euro,
        index=None,
        placeholder="Select euro...",
        disabled= not check_euro
        )
    if check_euro:
        filtering='by specify'

    if filtering == 'by extra time':
        st.dataframe(result.loc[result['Extra Time']==True])
    elif filtering == 'by penatlies':
        st.dataframe(result.loc[result['Penatly'] == True])
    elif check_euro and isinstance(option,str):
        st.dataframe(result.loc[result['Euro'].isin([option])])
    else:
        st.dataframe(result)

    result['H'] = result['H'].astype(int)
    result['A'] = result['A'].astype(int)
    result['SUM'] = result['H'] + result['A']
    euros = result.groupby('Euro').agg({'SUM': 'sum', 'H': 'count'})
    euros['AVG'] = round(euros['SUM'] / euros['H'], 2)
    euros.rename(columns={'H': 'COUNT'})

    stages = result.groupby('Stage').agg({'SUM': 'sum', 'H': 'count'})
    stages['AVG'] = round(stages['SUM'] / stages['H'], 2)
    stages.rename(columns={'H': 'COUNT'})
    order = ['Group', 'Round of 16', 'Quarter-finals', 'Semi-finals', '3rd place', 'Final']
    stages = stages.reindex(order)

    fig56(euros,stages)

with tab4:
    elo_start = pd.read_csv('/Users/admin/Downloads/data/ELO_start.csv', sep =';')
    elo_end = pd.read_csv('/Users/admin/Downloads/data/ELO_end.csv', sep =';')
    elo = pd.merge(elo_start, elo_end, on=['Euro', 'Team'], how='inner')
    elo = elo.reset_index(drop=True)
    elo.index += 1
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
    change={'Team':'Country','Rating_x':'ELO before','Local_x':'REB','Global_x':'RGB','Rating_y':'ELO after','Local_y':'REA','Global_y':'RGA','Change':'ELO change'}
    elo.rename(columns=change, inplace=True)
    if option:
        col1, col2 = st.columns(2)
        col1.metric('Best Euro:', int(elo.loc[(elo['ELO change'] == elo['ELO change'].loc[elo['Country'] == option].max()) & (elo['Country'] == option)]['Euro']))
        col2.metric('Worst Euro:', int(elo.loc[(elo['ELO change'] == elo['ELO change'].loc[elo['Country'] == option].min()) & (elo['Country'] == option)]['Euro']))
        elo = elo.loc[elo['Country']==option]
        st.dataframe(elo.style.highlight_max(subset = ['ELO change'],color = 'green', axis =0),use_container_width=True)
        fig7(elo)
    else:
        st.dataframe(elo, use_container_width=True)

with tab5:
    tabA, tabB, tabC, tabD, tabE, tabF = st.tabs(["Group A","Group B","Group C","Group D","Group E","Group F"])

    with tabA:
        link= '/Users/admin/Downloads/data/h2h/groupA_H2H_modified.csv'
        view_group(team_A,link)
    with tabB:
        link= '/Users/admin/Downloads/data/h2h/groupB_H2H_modified.csv'
        view_group(team_B,link)
    with tabC:
        link= '/Users/admin/Downloads/data/h2h/groupC_H2H_modified.csv'
        view_group(team_C,link)
    with tabD:
        link= '/Users/admin/Downloads/data/h2h/groupD_H2H_modified.csv'
        view_group(team_D,link)
    with tabE:
        link= '/Users/admin/Downloads/data/h2h/groupE_H2H_modified.csv'
        view_group(team_E,link)
    with tabF:
        link= '/Users/admin/Downloads/data/h2h/groupF_H2H_modified.csv'
        view_group(team_F,link)

with tab6:
    st.subheader('Passes for Country')
    passes = pd.read_csv('/Users/admin/Downloads/data/2024Q_passes.csv', sep=';')
    passes = pd.merge(passes, all_df, on=['Country'], how='left')
    passes['Euro 2024'].fillna(False, inplace=True)
    passes = passes.reset_index(drop=True)
    passes.index += 1
    st.dataframe(passes,column_config={
            "Pass Accuracy (avg)": st.column_config.ProgressColumn(
                "Pass Accuracy (avg)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            )})
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
    with st.sidebar:
        st.write("Bartosz Wajman")
