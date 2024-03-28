from bs4 import BeautifulSoup
import requests
import csv

class Player:
    def __init__(self, name, national, caps):
        self.name = name
        self.national = national
        self.caps = caps

for i in range(1, 51):
    url = f'https://www.worldfootball.net/alltime_top_player/em/{i}/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    row = []
    for i in soup.find(class_="standard_tabelle").find_all('td'):
        row.append(i.text.replace("\n", ""))
        if len(row) == 3:
            player=Player(row[0], row[1], row[2])
            with open('players.csv', 'a', newline='\n', encoding='utf-8') as file:
                wr = csv.writer(file)
                wr.writerow([player.name,player.national,player.caps])
            row = []
