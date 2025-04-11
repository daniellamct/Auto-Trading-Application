import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup

def check_valid(sy, sm, sd, ey, em, ed, stock):
    try:
        # stock validation
        stock_valid = False
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = f'https://finance.yahoo.com/quote/{stock}/profile/'
        response = requests.get(url, headers=headers)
        received_content = response.text
        soup = BeautifulSoup(received_content, 'html.parser')
        companyName_area = soup.find('section', class_='container yf-xxbei9 paddingRight')
        if companyName_area:
            cn_area = companyName_area.find('h1', class_='yf-xxbei9')
            if cn_area:
                stock_valid = True

        # Date Validation
        start_date = datetime(int(sy), int(sm), int(sd))
        end_date = datetime(int(ey), int(em), int(ed))

        return stock_valid and (start_date < end_date)
    
    except:
        return False


def check_stock(stock):
    try:
        # stock validation
        stock_valid = False
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = f'https://finance.yahoo.com/quote/{stock}/profile/'
        response = requests.get(url, headers=headers)
        received_content = response.text
        soup = BeautifulSoup(received_content, 'html.parser')
        companyName_area = soup.find('section', class_='container yf-xxbei9 paddingRight')
        if companyName_area:
            cn_area = companyName_area.find('h1', class_='yf-xxbei9')
            if cn_area:
                stock_valid = True

        return stock_valid
    
    except:
        return False


def check_num(fast, slow):
    if fast.isdigit() and slow.isdigit():
        return int(slow) > int(fast)
    return False


def check_date(sy, sm, sd, ey, em, ed):
    try:
        # Date Validation
        start_date = datetime(int(sy), int(sm), int(sd))
        end_date = datetime(int(ey), int(em), int(ed))

        return start_date < end_date
    
    except:
        return False