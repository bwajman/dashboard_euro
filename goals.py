from bs4 import BeautifulSoup
import requests
import csv

class Player:
    def __init__(self, id, name, national, caps, goals, penatly, avg):
        self.id = id
        self.name = name
        self.national = national
        self.caps = caps
        self.goals = goals
        self.penatly = penatly
        self.avg = avg

for i in range(1, 11):
    url = f'https://www.worldfootball.net/alltime_goalgetter/em/tore/{i}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    row = []
    for i in soup.find(class_="standard_tabelle").find_all('td'):
        row.append(i.text.replace("\n", ""))
        if len(row) == 7:
            player=Player(row[0], row[1], row[2], row[3], row[4], row[5], row[6])
            with open('goals.csv', 'a', newline='\n', encoding='utf-8') as file:
                wr = csv.writer(file)
                wr.writerow([player.id,player.name,player.national,player.caps,player.goals,player.penatly,player.avg])
            row = []

