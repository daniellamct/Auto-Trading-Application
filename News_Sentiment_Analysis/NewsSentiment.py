import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from summarizer import Summarizer

def fun_news(stock):

    try:
        # Load the pretrained model
        state = "Loading models"
        print("\n" + state)
        task = 'sentiment-latest'
        model_dir = f'News_Sentiment_Analysis/cardiffnlp/twitter-roberta-base-{task}'
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        # Load the summarizer
        model_summarizer = Summarizer()

        # Web Scraping
        state = "Gathering News Links"
        print(state)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        #stock = "AAPL"
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
        cur = 0
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
                    print("Skipped news: The content length is too short. ")
                    continue

                if(len(title) != 0 and len(paragraph) != 0):
                    # Summarize
                    paragraph_500 = model_summarizer(paragraph, ratio=0.8)
                    paragraph_100 = model_summarizer(paragraph, ratio=0.2)

                    # Sentiment Analysis by TextBlob (disabled to simplify the application)
                    result_TextBlob = " "
                    #analysis = TextBlob(f'{title}. {paragraph_500}')
                    #if(analysis.polarity > 0.2):
                    #    result_TextBlob = f'Positive (with {analysis.polarity:.2f} polarity score)'
                    #elif(analysis.polarity < -0.2):
                    #    result_TextBlob = f'Negative (with {analysis.polarity:.2f} polarity score)'
                    #else:
                    #    result_TextBlob = f'Neutral (with {analysis.polarity:.2f} polarity score)'


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

                    # Stop if processed enough news
                    cur += 1
                    if cur == 4:
                        break
                    

            except Exception as exception:
                print(f'Skipped')
                print(f'{exception}\n')

        # Result
        for i in range(0, len(titles)):
            print(f'\n\n{i+1}: {titles[i]}: ')
            #print(f'Rating (By TextBlob): {sentiments_Textblob[i]}')
            print(f'Rating (By Cardiffnlp): {sentiments_Cardiffnlp[i]}')
            #print(f'Origin Article: {paragraphs_500[i]}\n')
            #print(f'Summarization (Within 500 words): {paragraphs_500[i]}\n')
            #print(f'Summarization (Within 100 words): {paragraphs_100[i]}\n')

        return [titles,sentiments_Textblob,sentiments_Cardiffnlp,paragraphs,paragraphs_500,paragraphs_100]

    except:
        return []


def fun_news1(stock="1"):
    vre=[]

    for v1 in range(6):
        vre.append([str(v1)+"1",str(v1)+"2",str(v1)+"3",str(v1)+"4",str(v1)+"5"])

    return vre
