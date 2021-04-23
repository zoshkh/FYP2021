import pandas as pd
import requests
from bs4 import BeautifulSoup

#Iterate through the pages of LSE official site and scrap tickers

url = 'https://www.londonstockexchange.com/live-markets/market-data-dashboard/price-explorer?markets=MAINMARKET&categories=EQUITY&subcategories=1&showonlylse=true'
r = requests.get(url)
html = r.text

soup = BeautifulSoup(html)

table = soup.find('table', {"class": "full-width price-explorer-results"})
table_body = table.find('tbody')
rows = table_body.find_all('tr')
data = []
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append([ele for ele in cols if ele])
##

url = 'https://www.londonstockexchange.com/live-markets/market-data-dashboard/price-explorer?markets=MAINMARKET&categories=EQUITY&subcategories=1&showonlylse=true&page='


for page in range(60):

    print('---', page, '---')

    r = requests.get(url + str(page))
    html = r.text
    soup = BeautifulSoup(html)

    table = soup.find('table', {"class": "full-width price-explorer-results"})
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])
    
    
lst2 = [item[0] for item in data]