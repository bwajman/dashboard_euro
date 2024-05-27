Dashboard created on the occasion of **EURO 2024** including information about previous European championships.

Used:
* **bs4**
* **ipyvizzu**
* **jupyter notebook**
* **matplotlib**
* **pandas**
* **plotly**
* **scikit-learn**
* **streamlite**
  
Content:
🗓️ all-time table
👤 players
🏅 results
📈 ELO rank
🏆 EURO 2024
✅ qualifications
🤖 predict matches

The dashboard consists of 7 main tabs:

🗓️ historical summary of all countries ever participating in the European Championships. Filterable by EURO 2024 participants.

4 metrics:
* goals scored 
* matches played
* average goals scored per match
* number of participating teams

All-time table - for each country:
* information whether the country is a participant in EURO 2024
* number of participations in the European Championships
* number of total matches (won, drawn, lost)
* goals scored and goals lost and the difference in goal ratio
* number of points scored
* number of points per participation in the championship
* number of trophies won

A charts:

* a bar chart showing the number of goals scored and lost for each country
* a bar chart showing how many times a country participated in the championship, and a plotted points chart showing the average number of points scored per participation
* an animated bar chart showing the number of matches played at the championship by a given country, broken down by the number of matches won, tied and lost (detailed information on countries that won at least 45% of their matches, tied at least 40% of their matches and lost less than 25% of their matches)

👤 information on players participating in the championship. Possible filtering by active players and by selected national teams.

8 metrics:
* Number of footballers who appeared at the championships
* Nationality of the players who appeared most often
* How many players from a given nationality made the most appearances
* How many total appearances these players had

* Number of goals scored 
* Nationality of footballers who scored most often
* How many footballers most often scored from a given nationality
* How many total goals these footballers had

Interactive pie chart to select the most popular countries (from 2 to 15) in terms of:
number of footballers who performed
* total number of matches played by these players
* goals scored
* goals scored from penalties

A table showing all players ever appearing at the European Championships, along with information on the number of appearances, goals scored, goals scored on penalties, goal average, and whether the player is active.

A bar chart showing the number of appearances for each nationality and a dot chart showing the number of players for each nationality.

* The top scorers of the European Championships:
A bar chart showing the number of scoring kings per European championship, and a line chart showing the number of goals needed to become the top scorer
* Own goal scorers
A pie chart showing the ratio of goals scored by active players to players who have already completed their careers

🏅 results of all matches. Possible to filter by type of completed match by regulation time, overtime and penalty kicks or specific Euro.

2 metrics:

* Number of matches completed after overtime
* Number of matches completed after penalty kicks

A table containing all the results of the matches. Information about the Euro, the phase in which the match was played, date, time, teams and results (up to halftime, after regulation time and overtime)  

A line graph showing the number of goals scored at the Euros and a dot graph showing the average goals scored at the Euros.

A bar chart showing the matches at a particular stage of the championship and a scoring chart showing the average goals at those stages.

📈 ELO ranking for each team at the European Championships.

A chart showing the best and worst performance of the national team at a given Euro.

After selecting a specific country:

2 metrics:
* worst Euro
* best Euro
Table showing ELO ranking before the championship and after the championship. The last column shows the change. In green the best score is marked. The table also shows the Ranking before in Europe and the World and also after.

Bar chart showing the change in ELO ranking for the championship.


🏆 summary of all groups from EURO 2024. 

After selecting teams, information about:
* balance of results of direct matches played and goals
* this information presented on two pie charts
* categories of matches played against each other.

A table containing all results of direct matches. Information about the date, the result, the competition in which the match was played and the result.

✅ summary statistics from the qualification of all participating teams - it is possible to set up a filter on EURO 2024 participants.
* passes
table with averages - passes per match, accurate passes, passing accuracy and ball possession
scoring chart showing passing accuracy from the total number of passes made
* shots
a table with information on all shots, accurate, missed, blocked, from the penalty area, from the penalty area. The table also includes information on offsides and goals scored.
point chart where you can choose what should be on the x-axis and y-axis
* fouls
table with information on the number of fouls, number of yellow cards and red cards
point chart showing number of fouls from yellow cards

🤖 Prediction fixtures

A neural network algorithm - MLPClassifier - was used. The model was learned from historical data containing all matches from the European Championships. The learning data were the ELO ranking for two teams and the phase of the tournament. The data was scaled. 
