from selenium import webdriver
from bs4 import BeautifulSoup
from textblob import TextBlob

# Web Scraping
stock = "AAPL"
url = f'https://finance.yahoo.com/quote/{stock}/news/?p={stock}'
driver = webdriver.Chrome()  
driver.get(url)
file = driver.page_source
soup = BeautifulSoup(file, 'html.parser')

# Article links extraction
li_li = soup.find_all('li', class_='story-item')
links = []
for li in li_li:
    link = li.find('a', href=True)
    if link:
        links.append(link['href'])

titles = []
paragraphs = []
articles = []
sentiments = []
for i in range(0, len(links)):
    title = "" 
    paragraph = ""

    try:
        # Web Scraping for each article
        url = links[i]
        driver.get(url)
        file = driver.page_source
        soup = BeautifulSoup(file, 'html.parser')

        # Title Extraction
        title_area = soup.find('div', class_='cover-title')
        if title_area:
            title = title_area.text

        # Paragraphs Extraction
        text_area = soup.find('div', class_='atoms-wrapper')
        if text_area:
            p_all = text_area.find_all('p')
            for p in p_all:
                paragraph += p.text

        if(len(title) != 0 and len(paragraph) != 0):
            # Sentiment Analysis
            article = f'{title}. {paragraph}'
            analysis = TextBlob(article)

            titles.append(title)
            paragraphs.append(paragraph)
            articles.append(article)
            sentiments.append(analysis.sentiment.polarity)
    except Exception as exception:
        print(f'Skipped\n')

for i in range(0, len(titles)):
    print(f'\n{i+1}: {titles[i]}: ')
    print(f'Rating: {sentiments[i]}\n')

