import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from summarizer import Summarizer

# Load the pretrained model
state = "Loading models" # Update GUI application state
print(state)
task = 'sentiment-latest'
model_dir = f'News Sentiment Analysis/cardiffnlp/twitter-roberta-base-{task}'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Load the summarizer
model_summarizer = Summarizer()

# Web Scraping
state = "Gathering News Links" # Update GUI application state
print(state)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
stock = "AAPL"
url = f'https://finance.yahoo.com/quote/{stock}/news/?p={stock}'
response = requests.get(url, headers=headers)
received_content = response.text
soup = BeautifulSoup(received_content, 'html.parser')

# Article links extraction
li_li = soup.find_all('li', class_='story-item')
links = []
for li in li_li:
    link = li.find('a', href=True)
    if link:
        links.append(link['href'])

titles = []
paragraphs = []
paragraphs_500 = []
paragraphs_100 = []
sentiments_Textblob = []
sentiments_Cardiffnlp = []
for i in range(0, len(links)):
    title = "" 
    paragraph = ""

    try:
        state = f'Fetching News: {str(i+1)}' # Update GUI application state
        print(state)

        # Web Scraping for each article
        url = links[i]
        response = requests.get(url, headers=headers)
        received_content = response.text
        soup = BeautifulSoup(received_content, 'html.parser')

        state = f'Processing News: {str(i+1)}' # Update GUI application state
        print(state)

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
        if len(paragraph) < 1000:
            print("Skipped news: The Content length is too short. ")
            continue

        if(len(title) != 0 and len(paragraph) != 0):
            # Summarize
            paragraph_500 = model_summarizer(paragraph, ratio=0.8)
            paragraph_100 = model_summarizer(paragraph, ratio=0.2)

            # Sentiment Analysis (TextBlob)
            analysis = TextBlob(f'{title}. {paragraph}')
            if(analysis.sentiment.polarity > 0.2):
                result_TextBlob = f'Positive (with {analysis.sentiment.polarity:.2f} polarity score)'
            elif(analysis.sentiment.polarity < -0.2):
                result_TextBlob = f'Negative (with {analysis.sentiment.polarity:.2f} polarity score)'
            else:
                result_TextBlob = f'Neutral (with {analysis.sentiment.polarity:.2f} polarity score)'

            # Sentiment Analysis (Cardiffnlp)
            encoded_input = tokenizer(f'{title}. {paragraph_500}', return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            if(scores[0] > scores[1]) and (scores[0] > scores[2]):
                # Positive
                result_Cardiffnlp = f'Positive (with {scores[0]*100:.2f}% probability)'
            elif(scores[1] > scores[0]) and (scores[1] > scores[2]):
                # Neutral
                result_Cardiffnlp = f'Neutral (with {scores[1]*100:.2f}% probability)'
            else:
                # Negative
                result_Cardiffnlp = f'Negative (with {scores[2]*100:.2f}% probability)'

            titles.append(title)
            paragraphs.append(paragraph)
            paragraphs_100.append(paragraph_100)
            paragraphs_500.append(paragraph_500)
            sentiments_Textblob.append(result_TextBlob)
            sentiments_Cardiffnlp.append(result_Cardiffnlp)
            # Update the GUI here

    except Exception as exception:
        print(f'Skipped')
        print(f'{exception}\n')

# These contents below will be shown in the GUI
for i in range(0, len(titles)):
    print(f'\n\n{i+1}: {titles[i]}: ')
    print(f'Rating (By TextBlob): {sentiments_Textblob[i]}')
    print(f'Rating (By Cardiffnlp): {sentiments_Cardiffnlp[i]}')
    #print(f'Origin Article: {paragraphs_500[i]}\n')
    #print(f'Summarization (Within 500 words): {paragraphs_500[i]}\n')
    #print(f'Summarization (Within 100 words): {paragraphs_100[i]}\n')
