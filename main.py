import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from ipyvizzu import Data, Config, Style, Chart
from ipyvizzustory import Story
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def create_chart(df):

    data = Data()

    df = pd.melt(df, id_vars=['Country'], value_vars=['W', 'D', 'L'], var_name='RESULT', value_name='COUNT')
    df['Total'] = df.groupby('Country')['COUNT'].transform('sum')
    df['Percentage'] = (df['COUNT'] / df['Total']) * 100
    df['Percentage'] =  df['Percentage'].round()
    df = df.sort_values(by=['Country', 'RESULT'], ascending=[False, False])

    data.add_data_frame(df)
    story = Story(data=data)
    story.set_size(800, 800)
    label_handler_method = "if(event.data.text.split(' ')[0] < 5) event.preventDefault()"

    story.add_event("plot-marker-label-draw", label_handler_method)
    chart = Chart(display="manual")
    chart.animate(data)
    chart.animate(
            Config(dict(x=["COUNT", "RESULT"], y=["Country"], color="RESULT", label="COUNT", align="stretch",
                        title="Matches for countries divided by results", lightness="RESULT",
                        legend="color")),
            Style({"logo": {"width": "250em"},
                   "plot": {"xAxis": {"title": {"color": "#00000000"}}, "paddingLeft": "2.5em",
                            "marker": {"colorPalette": "  #298505 #d1120f #0992ad",
                                       "minLightness": 0, "maxLightness": 0.4}
                            }}),duration=1
        )
    
    chart.animate(Config({
                "split": True,"align": "min", "title": "Matches for countries categorized by results."
        }),
            Style({"plot": {"xAxis": {"label":{"color": "#00000000"}}
            }}),duration=2, delay=5,
        )

    chart.animate(Data.filter("record.RESULT == 'W' && record.Percentage >= 45"), Config(
            {
                "split": False, "align": "min", "title": "Countries that won at least 45% of their matches."
            }),
             Style({"plot": {"xAxis": {"label": {"color": "#00000000"}}
                             }}),duration=2, delay=5,
    )

    chart.animate(Data.filter("record.RESULT == 'D' && record.Percentage >= 40"), Config(
            {
                "split": False, "align": "min", "title": "Countries that have drawn at least 40% of their matches."
            }),
             Style({"plot": {"xAxis": {"label": {"color": "#00000000"}}
                             }}),duration=2, delay=5,
             )

    chart.animate(Data.filter("record.RESULT == 'L' && record.Percentage <= 25"), Config(
            {
                "split": False, "align": "min", "title": "Countries that lost less than 25% of their matches"
            }),
             Style({"plot": {"xAxis": {"label": {"color": "#00000000"}}
                             }}),duration=2, delay=5,
             )

    return chart._repr_html_()

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
        title='Matches and players per country',
        xaxis_tickfont_size=12,
        yaxis=dict(
            title='Matches / Players',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    return st.plotly_chart(fig4)

def fig56(df, df1):
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
    fig6.update_layout(xaxis_title='Stage', title='Matches per stage and avg goals/stage')

    return st.plotly_chart(fig5), st.plotly_chart(fig6)

def fig7(df):
    fig7 = go.Figure()

    colors = ['red' if val < 0 else 'green' for val in df['ELO change']]

    fig7.add_trace(go.Bar(
        x=df['Euro'],
        y=df['ELO change'],
        name='Goals',
        yaxis='y',
        marker=dict(color=colors) 
    ))

    fig7.update_xaxes(tickmode='array', tickvals=df['Euro'])
    fig7.update_layout(xaxis_title='EURO', yaxis_title='ELO points', title='Changes in ELO points')

    return st.plotly_chart(fig7)

def fig8(df):
    fig8 = go.Figure()

    colors = [1 if active else 0 for active in df['Active']]

    fig8.add_trace(go.Scatter(
        x=df['Matches'].head(20),
        y=df['Goals'].head(20),
        mode='markers',
        hoverinfo='text',
        text=df['Player'] + '<br>' + 'Matches: ' + df['Matches'].astype(str) + '<br>' + 'Goals: ' + df['Goals'].astype(
            str) + ' (' + df['Penatlies'].astype(str) + ') ' + '<br>' + 'AVG: ' + df['AVG'].astype(str),
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


def fig9(W, D, L):
    fig9 = go.Figure()

    fig9.add_trace(go.Pie(
        labels=['W', 'D', 'L'],
        values=[W, D, L],
        hole=0.3
    ))

    fig9.update_layout(title='Change ELO rank in EURO')

    st.plotly_chart(fig9)


def fig10(df):
    fig10 = go.Figure()
    unique_euros = df['Euro'].unique()
    players_count = {euro: df[df['Euro'] == euro]['Player'].count() for euro in unique_euros}

    fig10.add_trace(
        go.Scatter(x=df['Euro'], y=df['Goals'], mode='lines+markers', name='Goals', yaxis='y', hovertext=df['Goals']))
    fig10.add_trace(
        go.Bar(x=list(players_count.keys()), y=list(players_count.values()), name='Players', marker_color='blue'))
    fig10.update_layout(xaxis_title='EURO', yaxis_title='Goals / Players', title='Topscorers in EURO')
    fig10.update_xaxes(tickmode='array', tickvals=df['Euro'])
    return st.plotly_chart(fig10)

def fig11(df):
    fig11 = go.Figure()

    fig11.add_trace(go.Bar(x=df['Euro'], y=df['ELO change best'], name='Best', yaxis='y', hovertext=df['Country best']))
    fig11.add_trace(go.Bar(x=df['Euro'], y=df['ELO change worst'], name='Worst', yaxis='y', hovertext=df['Country worst']))
    fig11.update_layout(xaxis_title='EURO', yaxis_title='ELO points', title='Best and worst performances at the EURO according to ELO rankings')
    fig11.update_xaxes(tickmode='array', tickvals=df['Euro'])

    return st.plotly_chart(fig11)

def pie_chart(x, y, desc):
    fig9 = go.Figure()
    fig9.add_trace(go.Pie(
        labels=x,
        values=y,
        hole=0.3
    ))
    fig9.update_layout(title=desc)
    st.plotly_chart(fig9)

def chart_passes(df):
    fig4 = go.Figure()

    colors = [1 if val else 0 for val in df['Euro 2024']]
    markers = [int(i.strip('%')) / 3.5 for i in df['Possession']]

    fig4.add_trace(go.Scatter(
        x=df['Pass Accuracy (avg)'],
        y=df['Total passes (avg)'],
        mode='markers',
        hoverinfo='text',
        text=df["Country"] + ' with possession ' + df["Possession"],
        name='all',
        marker=dict(
            color=colors,
            colorscale='Viridis',
            size=markers
        )
    ))
    fig4.update_layout(
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            gridwidth=1
        )
    )

    fig4.update_layout(
        title='Goals and matches',
        xaxis=dict(title='Pass Accuracy (avg)', titlefont_size=16, tickfont_size=14),
        yaxis=dict(title='Total passes (avg)', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )

    return st.plotly_chart(fig4)


def chart_shots(df):
    option = st.selectbox(
        "Which stats should be on x axes?",
        df.columns[1:-1],
        index=0,
        placeholder="Select euro...",
    )
    option1 = st.selectbox(
        "Which stats should be on y axes?",
        df.columns[1:-1],
        index=1,
        placeholder="Select euro...",
    )

    fig5 = go.Figure()

    colors = [1 if val else 0 for val in df['Euro 2024']]

    fig5.add_trace(go.Scatter(
        x=df[option],
        y=df[option1],
        mode='markers',
        hoverinfo='text',
        text=df['Country'],
        name='all',
        marker=dict(
            color=colors,
            colorscale='Viridis'
        )
    ))

    fig5.update_layout(
        title=f'{option} vs {option1}',
        xaxis=dict(title=option, titlefont_size=16, tickfont_size=14),
        yaxis=dict(title=option1, titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )

    return st.plotly_chart(fig5)


def chart_fouls(df):
    fig6 = go.Figure()

    colors = [1 if val else 0 for val in df['Euro 2024']]
    markers = [int(i) + 10 for i in df['Red cards']]

    fig6.add_trace(go.Scatter(
        x=df['Fools'],
        y=df['Yellow cards'],
        mode='markers',
        hoverinfo='text',
        text=df['Country'],
        name='all',
        marker=dict(
            color=colors,
            colorscale='Viridis',
            size=markers
        )
    ))

    fig6.update_layout(
        title='Fouls / yellow cards',
        xaxis=dict(title='Fouls', titlefont_size=16, tickfont_size=14),
        yaxis=dict(title='Yellow cards', titlefont_size=16, tickfont_size=14),
        legend=dict(x=0, y=1.0, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'),
        bargap=0.2,
        bargroupgap=0.1
    )

    st.plotly_chart(fig6)

def most_appear(df):
    count = df['Player'].count()
    nation = df.groupby(by='Country').count().sort_values('Player', ascending=False).index[0]
    player = df.groupby(by='Country').count().sort_values('Player', ascending=False)['Player'][0]
    caps = df.loc[df['Country'] == nation, 'Matches'].sum()
    return count, nation, player, caps

def most_goals(df):
    if len(df) > 0:
        gplayers = df['Player'].count()
        ggoals = int(df['Goals'].sum())
        gnation = df.groupby(by='Country')[['Country', 'Goals']].sum().sort_values(by='Goals', ascending=False).index[0]
        gplayer = df.loc[((df['Country'] == gnation) & (df['Goals']>0))]['Player'].count()
        gcaps = int(df.loc[df['Country'] == gnation, 'Goals'].sum())
        return gplayers, ggoals, gnation, gplayer, gcaps

def view_group(team, link):
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox(
            'Choose country to compare?',
            team)
    with col2:
        team.remove(home)
        away = st.selectbox(
            'Choose country to compare?',
            team)

    A = pd.read_csv(link, sep=',')
    st.write(f'You selected: {home} vs {away}')
    A.drop(A.columns[0], axis=1, inplace=True)
    A = A.reset_index(drop=True)
    A.index += 1
    A = A.loc[(A['Home'].str.contains(home) & (A['Away'].str.contains(away))) | (
                (A['Home'].str.contains(away)) & (A['Away'].str.contains(home)))]
    H = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'H')) | (
                (A['Home'].str.contains(away)) & (A['Result'] == 'A'))].count()
    D = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'D')) | (
                (A['Home'].str.contains(away)) & (A['Result'] == 'D'))].count()
    L = A['Result'].loc[((A['Home'].str.contains(home)) & (A['Result'] == 'A')) | (
                (A['Home'].str.contains(away)) & (A['Result'] == 'H'))].count()
    GF = A['H'].loc[A['Home'].str.contains(home)].sum() + A['A'].loc[A['Away'].str.contains(home)].sum()
    GA = A['A'].loc[A['Home'].str.contains(home)].sum() + A['H'].loc[A['Away'].str.contains(home)].sum()
    st.caption(f'{home} had {H} wins {D} draws and {L} defeats against {away} scoring {GF} goals and losing {GA} goals.')

    comp = [(value, key) for value, key in A["Competitions"].value_counts().items()]
    comp_str = ", \n".join([f"{value2} - {key2}" for (value2, key2) in comp])
    st.caption(f'{home} has played with {away} {A["Result"].count()} times in {comp_str}')
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

st.subheader("Everything you have to know about the coming EURO! 🥳")

team_A = ['Germany', 'Scotland', 'Switzerland', 'Hungary']
team_B = ['Spain', 'Italy', 'Croatia', 'Albania']
team_C = ['Denmark', 'England', 'Slovenia', 'Serbia']
team_D = ['France', 'Poland', 'Netherlands', 'Austria']
team_E = ['Ukraine', 'Romania', 'Slovakia', 'Belgium']
team_F = ['Portugal', 'Turkey', 'Czech Republic', 'Georgia']

euro = [element for sublist in [team_A, team_B, team_C, team_D, team_E, team_F] for element in sublist]
euro2024df = pd.DataFrame(sorted(euro), columns=['Country'])
euro2024df['Euro 2024'] = True

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🗓️ all-time table", "👤 players", "🏅 results", "📈 ELO rank", "🏆 EURO 2024", "✅ qualifications",'🤖 **:red[predict matches]**'])

with tab1:
    all = pd.read_csv('data/all.csv', sep=';')
    df = pd.read_csv('data/all.csv', sep=';')
    all.loc[all['Country'] == 'Türkiye', 'Country'] = 'Turkey'
    df.loc[df['Country'] == 'Türkiye', 'Country'] = 'Turkey'
    all = pd.merge(all, euro2024df, on=['Country'], how='left')
    df = pd.merge(df, euro2024df, on=['Country'], how='left')
    all['Euro 2024'].fillna(False, inplace=True)
    df['Euro 2024'].fillna(False, inplace=True)

    last_column = all.pop(all.columns[-1])
    all.insert(2, last_column.name, last_column)
    all.set_index([all.columns[0], all.columns[1]], inplace=True)
    all.index.names = ['No', 'Country']
    all['Points per tournament'].fillna(0, inplace=True)
    all['Points per tournament'] = all['Points per tournament'].round(2).apply(lambda x: '{:.2f}'.format(x))

    trophy = lambda x: (x * '🏆')
    all['Trophy'] = all['Trophy'].apply(trophy)

    euro2024 = st.toggle('Euro 2024')
    col1, col2, col3, col4 = st.columns(4)
    if euro2024:
        all = all.loc[all['Euro 2024'] == True]
        df = df.loc[df['Euro 2024'] == True]
        st.caption('* 2 of the 24 countries that will play at EURO 2024 have never played at a EURO before - Georgia and Serbia (which played as Yugoslavia in EURO 2020).')

    matches = int(all['Matches'].sum() / 2)
    col1.metric('Goals', all['GF'].sum())
    col2.metric('Matches', matches)
    col3.metric('Avg goals/match', round(all['GF'].sum() / matches, 2))
    col4.metric('Teams', len(all.index.get_level_values(1)))

    st.dataframe(
        all,
        column_config={
            "Points": st.column_config.ProgressColumn(
                "Points",
                format="%f",
                min_value=0,
                max_value=94,
            ),
            "Points per tournament": st.column_config.ProgressColumn(
                "Points per tournament",
                format="%.2f",
                min_value=0,
                max_value=8.38,
            )
        }, use_container_width=True
    )

    st.caption("W - Wins \ D - Draws \ L - Losses \ GF - Goals for \ GA - Goals against \ GD - Goals difference")

    teams = all.sort_index(level='Country')
    
    fig1()
    fig2()

    if st.button('Animation', type='primary'):
        CHART = create_chart(df)
        html(CHART, width=800, height=800)

with tab2:
    players = pd.read_csv('data/players_modified.csv', sep=',', index_col=0)
    goals = pd.read_csv('data/goals_modified.csv', sep=',', index_col=1)
    top = pd.read_csv('data/topscorer_modified.csv', sep=';', index_col=0)
    own_goals = pd.read_csv('data/all_own_goals_modified.csv', sep=';', index_col=0)

    players = pd.merge(players, goals, on=['Player'], how='left')
    players.drop(columns=players.columns[4:7], axis=1, inplace=True)
    players.drop(columns=players.columns[-1], axis=1, inplace=True)
    players.fillna(0, inplace=True)
    players = players.reset_index(drop=True)
    players.index += 1
    players['AVG'] = players['AVG'].round(2).apply(lambda x: '{:.2f}'.format(x))

    rename = {'Matches_x': 'Matches', 'Active_x': 'Active', 'Country_x': 'Country'}
    players = players.rename(columns=rename)
    order = ['Player', 'Country', 'Matches', 'Goals', 'Penatlies', 'AVG', 'Active']
    players = players[order]

    teams = sorted(players['Country'].unique())
    cols = st.columns([1, 3])
    on = cols[0].toggle('Active player')
    select = cols[1].checkbox('Select teams')
    options = cols[1].multiselect(
        'Which country do you want to check?',
        teams,
        teams[24],
        disabled=not select)

    if on and select and len(options) > 0:
        players = players.loc[(players['Active'] == True) & (players['Country'].isin(options))]
    elif on:
        players = players.loc[players['Active'] == True]
    elif select and len(options) > 0:
        players = players.loc[players['Country'].isin(options)]
    else:
        players = players

    try:
        count, nation, player, caps = most_appear(players)
        gplayers, ggoals, gnation, gplayer, gcaps = most_goals(players)
        col1, col2, col3, col4 = st.columns(4)
        if len(options) == 1 and select:
            col1.metric('Players', player)
            col2.metric('apperances', caps)
            col3.metric('Players', gplayer)
            col4.metric('goals', gcaps)
        else:
            col1.metric('Total players', count)
            col2.metric('Most popular nation is', nation)
            col3.metric('with players', player)
            col4.metric('and apperances', caps)
            col1.metric('Total Goals', ggoals)
            col2.metric('Most popular nation is', gnation)
            col3.metric('with players', gplayer)
            col4.metric('and goals', gcaps)
        if select and len(options) >= 2:
            with st.expander('📊 statistics for selected countries:'):
                a = st.slider('How many countries do you want to show on the chart?', 2, 15, 5)
                option = st.selectbox(
                    "Which stats do you interest?",
                    ['Player', 'Matches', 'Goals', 'Penatlies'],
                    index=0,)
                if option == 'Player':
                    x = players.groupby(by='Country')[option].count().sort_values(ascending=False)[:a]
                else:
                    x = players.groupby(by='Country')[option].sum().sort_values(ascending=False)[:a]
                pie_chart(x.index, x, f'{option} group by country')
        st.dataframe(players, use_container_width=True)
        count, nation, player, caps = most_appear(players)
        with st.expander("📊 show data of chart"):
            fig4(players)
    except:
        st.error('No records to display')

    with st.expander("👑 topscorers"):
        top = top.reset_index(drop=True)
        top.index += 1
        top['Euro'] = top['Euro'].astype(str)
        st.dataframe(top, use_container_width=True)
        fig10(top)

    with st.expander('😥 own goals:'):
        own_goals = own_goals.reset_index(drop=True)
        own_goals.index += 1
        st.dataframe(own_goals, use_container_width=True)
        pie_chart(own_goals['Active'], own_goals['Goals'], 'Goals scored by active players')

with tab3:
    result = pd.read_csv('data/result_modified.csv', sep=',', index_col=0)
    col1, col2 = st.columns(2)
    col1.metric('Matches that ended in extra time', result['Extra Time'].sum())
    col2.metric('Matches that were ended by penalties.', result['Penatly'].sum())

    result['Euro'] = result['Euro'].astype(str)
    euro = sorted(result['Euro'].unique())

    with col2:
        check_euro = st.checkbox('check Euro')
    with col1:
        filtering = st.radio(
            "Select matches:",
            ["no specify", "by extra time", "by penatlies"],
            key="visibility",
            disabled=check_euro
        )
    with col2:
        option = st.selectbox(
            "Which euro do you want to check?",
            euro,
            index=None,
            placeholder="Select euro...",
            disabled=not check_euro
        )
    if check_euro:
        filtering = 'by specify'
    if filtering == 'by extra time':
        st.dataframe(result.loc[result['Extra Time'] == True])
    elif filtering == 'by penatlies':
        st.dataframe(result.loc[result['Penatly'] == True])
    elif check_euro and isinstance(option, str):
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

    fig56(euros, stages)

with tab4:
    elo_start = pd.read_csv('data/ELO/ELO_start.csv', sep=';')
    elo_end = pd.read_csv('data/ELO/ELO_end.csv', sep=';')
    elo = pd.merge(elo_start, elo_end, on=['Euro', 'Team'], how='inner')
    elo = elo.reset_index(drop=True)
    elo.index += 1

    elo['Change'] = elo['Rating_y'] - elo['Rating_x']
    elo['Euro'] = elo['Euro'].astype(str)
    order = ['Euro', 'Team', 'Rating_x', 'Local_x', 'Global_x', 'Rating_y', 'Local_y', 'Global_y', 'Change']
    elo = elo[order]
    change = {'Team': 'Country', 'Rating_x': 'ELO before', 'Local_x': 'REB', 'Global_x': 'RGB', 'Rating_y': 'ELO after',
              'Local_y': 'REA', 'Global_y': 'RGA', 'Change': 'ELO change'}
    elo.rename(columns=change, inplace=True)

    best = elo.groupby(by='Euro')['ELO change'].max()
    best = pd.DataFrame(best)
    best['Euro_1'] = best.index
    worst = elo.groupby(by='Euro')['ELO change'].min()
    worst = pd.DataFrame(worst)
    worst['Euro_1'] = worst.index
    elo_bw = pd.merge(best, worst, on=['Euro_1'], how='inner')
    rename = {'ELO change_x': 'ELO change', 'Euro_1': 'Euro'}
    elo_bw.rename(columns=rename, inplace=True)
    elo_bw = pd.merge(elo_bw, elo, on=['ELO change', 'Euro'])
    rename = {'ELO change': 'ELO change best', 'ELO change_y': 'ELO change'}
    elo_bw.rename(columns=rename, inplace=True)
    elo_bw = pd.merge(elo_bw, elo, on=['ELO change', 'Euro'])
    rename = {'ELO change': 'ELO change worst', 'Country_x': 'Country best', 'Country_y': 'Country worst'}
    elo_bw.rename(columns=rename, inplace=True)

    fig11(elo_bw)

    team = sorted(elo['Country'].unique())
    option = st.selectbox(
        "Which country are you interested in?",
        team,
        index=None,
        placeholder="Select country...",
    )

    if option:
        col1, col2 = st.columns(2)
        col1.metric('Best Euro:', int(elo.loc[(elo['ELO change'] == elo['ELO change'].loc[
            elo['Country'] == option].max()) & (elo['Country'] == option)]['Euro']))
        col2.metric('Worst Euro:', int(elo.loc[(elo['ELO change'] == elo['ELO change'].loc[
            elo['Country'] == option].min()) & (elo['Country'] == option)]['Euro']))
        elo = elo.loc[elo['Country'] == option]
        st.dataframe(elo.style.highlight_max(subset=['ELO change'], color='green', axis=0), use_container_width=True)
        st.caption('* REB - Rank Europe Before / RGB - Rank Global Before / REA - Rank Europe After / RGA - Rank Global After')
        fig7(elo)
    else:
        st.dataframe(elo, use_container_width=True)
        st.caption('* REB - Rank Europe Before / RGB - Rank Global Before / REA - Rank Europe After / RGA - Rank Global After')

with tab5:
    tabA, tabB, tabC, tabD, tabE, tabF = st.tabs(["Group A", "Group B", "Group C", "Group D", "Group E", "Group F"])

    with tabA:
        link = 'data/h2h/groupA_H2H_modified.csv'
        view_group(team_A, link)
    with tabB:
        link = 'data/h2h/groupB_H2H_modified.csv'
        view_group(team_B, link)
    with tabC:
        link = 'data/h2h/groupC_H2H_modified.csv'
        view_group(team_C, link)
    with tabD:
        link = 'data/h2h/groupD_H2H_modified.csv'
        view_group(team_D, link)
    with tabE:
        link = 'data/h2h/groupE_H2H_modified.csv'
        view_group(team_E, link)
    with tabF:
        link = 'data/h2h/groupF_H2H_modified.csv'
        view_group(team_F, link)

with tab6:
    tab6_active = st.toggle('Euro 2024', key='tab6_active')

    passes = pd.read_csv('data/qualification/2024Q_passes.csv', sep=';')
    shots = pd.read_csv('data/qualification/2024Q_shots.csv', sep=';')
    offside = pd.read_csv('data/qualification/data/2024Q_offsides.csv', sep=';')
    goals = pd.read_csv('data/qualification/data/2024Q_goals.csv', sep=';')
    possession = pd.read_csv('data/qualification/2024Q_possession.csv', sep=';')

    passes.drop(columns=['Total passes', 'Passes arrived'], axis=1, inplace=True)
    passes = passes.sort_values(by='Total passes (avg)', ascending=False)
    possession.drop(possession.columns[0], axis=1, inplace=True)
    passes = pd.merge(passes, possession, on=['Country'], how='left')
    passes = pd.merge(passes, euro2024df, on=['Country'], how='left')
    passes['Euro 2024'].fillna(False, inplace=True)
    passes = passes.reset_index(drop=True)
    passes.index += 1
    goals.drop(columns=['#'], inplace=True, axis=0)
    offside.drop(columns=['Index'], inplace=True, axis=0)
    shots = pd.merge(shots, offside, on=['Country'], how='left')
    shots = pd.merge(shots, goals, on=['Country'], how='left')
    shots = pd.merge(shots, euro2024df, on=['Country'], how='left')
    shots['Euro 2024'].fillna(False, inplace=True)
    shots.drop(
        columns=['Total shots (avg)', 'Shots on target (avg)', 'Shots off target (average)', 'Shots blocked (average)',
                 'Shots in box (avg)', 'Shots outside box (avg)', 'Offside(avg)'], axis=1, inplace=True)
    shots = shots.reset_index(drop=True)
    shots.index += 1
    fouls = pd.read_csv('data/qualification/2024Q_fouls.csv', sep=';')
    fouls.drop(columns=['#', 'Fools (avg)'], inplace=True, axis=0)
    yellow_cards = pd.read_csv('data/qualification/2024Q_yellow_cards.csv', sep=';')
    yellow_cards.drop(columns=['#'], inplace=True, axis=0)
    red_cards = pd.read_csv('data/qualification/2024Q_red_cards.csv', sep=';')
    red_cards.drop(columns=['#'], inplace=True, axis=0)
    cards = pd.merge(fouls, yellow_cards, on=['Country'], how='left')
    cards = pd.merge(cards, red_cards, on=['Country'], how='left')
    cards['Red cards'].fillna(0, inplace=True)
    cards = pd.merge(cards, euro2024df, on=['Country'], how='left')
    cards['Euro 2024'].fillna(False, inplace=True)
    cards = cards.reset_index(drop=True)
    cards.index += 1

    if tab6_active:
        passes = passes.loc[passes['Euro 2024'] == True]
        shots = shots.loc[shots['Euro 2024'] == True]
        cards = cards.loc[cards['Euro 2024'] == True]

    with st.expander('passing in qualifications'):
        cmap = plt.colormaps['RdYlGn']
        st.dataframe(passes.style.background_gradient(cmap=cmap, subset=['Total passes (avg)', 'Passes arrived (avg)'],
                                                      vmin=(95), vmax=700, axis=None))
        chart_passes(passes)
    st.divider()
    with st.expander('shooting in qualifications'):
        st.dataframe(shots, use_container_width=True)
        chart_shots(shots)
    st.divider()
    with st.expander('fouls in qualifications'):
        st.dataframe(cards, use_container_width=True)
        chart_fouls(cards)

with tab7:
    with st.expander('🎲 information about model'):
        st.write("""The *neural network algorithm* used was the **MLPClassifier**. The model was learned from historical data (*trained by 269 matches : tested by 68 matches / 0.8 : 0.2*) containing all matches from the European Championships. The learning data was the ***ELO ranking*** (before the start of the championship) for two teams and the ***phase of the tournament***. The data was scaled because the ELO ranking in the past was lower than at present.
        ***The maximum iteration was set at 1000***. The optimal result - an efficiency of 63,24% - was obtained with ***100 hidden layers***, but this can be changed with a slider.""")

        st.caption('Remember:  The result is not staggering but this is sport and a lot of random factors determine the final result.')

    euro_2024_df_start = elo_start.loc[elo_start['Euro']==2024]
    euro_2024_df_start.drop(columns=['Euro','Local','Global'], axis=1, inplace=True)
    euro_2024_df_start = euro_2024_df_start.reset_index(drop=True)

    team_A = ['Germany', 'Scotland', 'Switzerland', 'Hungary']
    team_B = ['Spain', 'Italy', 'Croatia', 'Albania']
    team_C = ['Denmark', 'England', 'Slovenia', 'Serbia']
    team_D = ['France', 'Poland', 'Netherlands', 'Austria']
    team_E = ['Ukraine', 'Romania', 'Slovakia', 'Belgium']
    team_F = ['Portugal', 'Turkey', 'Czech Republic', 'Georgia']
    teams = sorted([element for sublist in [team_A, team_B, team_C, team_D, team_E, team_F] for element in sublist])

    stage = {'Group':0.0, 'Round of 16': 0.2, 'Quarter-final': 0.4, 'Semi-final': 0.6, 'Final': 1.0}
    col1, col2, col3 = st.columns(3)
    with col1:
        home = st.selectbox(
            'Choose country to match?',
            teams)
    with col2:
        teams.remove(home)
        away = st.selectbox(
            'Choose country to match?',
            teams)
    with col3:
        stage_form = st.selectbox(
            'Select stage?',
            stage)

    abcde = st.slider('Select hidden layers of model?', 20, 150, 100)

    scaler = MinMaxScaler()
    test = pd.read_csv('data/ML/ML_elo.csv',index_col=0)
    rescaled = scaler.fit_transform(test.iloc[:, :3])
    rescaled_2 = scaler.fit_transform(euro_2024_df_start.iloc[:, -1:])
    scaled_2024 = pd.Series(rescaled_2[:,0], name="Scaled")
    home_index_target_value = euro_2024_df_start[euro_2024_df_start['Team'] == home].index[0]
    rank_home = scaled_2024.iloc[home_index_target_value]
    away_index_target_value = euro_2024_df_start[euro_2024_df_start['Team'] == away].index[0]
    rank_away = scaled_2024.iloc[away_index_target_value]

    def determine_result(row):
        if row['H'] > row['A']:
            return 'H'
        elif row['H'] == row['A']:
            return 'D'
        else:
            return 'A'

    test['Result'] = test.apply(determine_result, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(rescaled, test.iloc[:, -1], test_size=0.2, random_state=42)
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(abcde,), max_iter=1000, random_state=42)
    mlp_classifier.fit(X_train, y_train)

    y_pred = mlp_classifier.predict(X_test)
    proba = mlp_classifier.predict_proba([[rank_home, rank_away, stage[stage_form]]])
    result = mlp_classifier.predict([[rank_home, rank_away, stage[stage_form]]])
    accuracy = accuracy_score(y_test,y_pred)

    if st.button('Predict'):
        st.success(f'The model has an accuracy of {accuracy*100:.2f}%, and it predicts with a probability of {max(proba[0])*100:.2f}% that the result of this match will be {result[0]}.', icon="✅")
        st.caption('H - Home win, D - draw, A - Away win')

with st.sidebar:
    st.subheader('Bartosz Wajman')
    st.caption("💼 [LinkedIn](https://www.linkedin.com/in/bartoszwajman)")
    st.write("📑 Dashboard created for EURO 2024 and past championships, featuring lots of visualizations and interaction possibilities. 🤖 A machine learning model was implemented to predict the performance of selected teams.")
    st.caption('''🪄 Libraries and tools used:  
               * bs4  
               * ipyvizzu  
               * jupyter notebook  
               * matplotlib  
               * pandas  
               * plotly  
               * scikit-learn  
               * streamlit  ''')
