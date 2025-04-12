# Stock Analysis Application

## Overview
This is a stock analysis application that involves techniques including machine learning, web scraping, and news sentiment analysis to help users analyze stocks. An executable GUI has been released and is available for free download.

## Language
This application is developed in Python

## Usage
There are 6 pages in the application

1. Introduction Page
- ![](/README_source/Introduction_page.png)
- This page provides a brief introduction on all the pages in the application

2. Graphs Analysis
- ![](/README_source/GA_page.png)
- Users can input their desired stock, date range, and select the desired moving averages
- 3 graphs will be generated
    - Closing Price Line Chart
    - Volume Line Chart
    - Daily Price Movement Violin Plot
- Some further descriptions are labelled next to each graph

3. Company Profile
- ![](/README_source/CP_page.png)
- Users can input their target stock, then the application will fetch the company profile information, including:
    - Company name
    - Sector
    - Industry
    - Location
    - Phone
    - URL
    - Description

4. News Analysis
- ![](/README_source/NA_page.png)
- User can input their target stock, then the application will fetch the latest news of the stock in the internet
- Sentiment analysis tool that are developed with CardffNLP will performance news sentiment analysis on the news 
- A summarization function is implemented. It allows users to read the news in original length, summarized 500 words, or summarized 100 words format
    
5. Price Prediction
- ![](/README_source/PP_page.png)
- User can input their target stock, then the application will perform price prediction with 2 trained machine learning models, ANN and LSTM
- Prediction performance evaluations will also be shown

6. Strategy Backtesting
- ![](/README_source/SB_page.png)
- User can input their desired stock, date range, and Moving Average lines
- The application will do a simulation to buy the stock when there are Bullish Crossovers, and sell the stock when there are Bearish Crossovers
- A result will be generated for evaluation


# Deployment Guidelines

## Overview
This section provides guidelines on deploying the application. The guidelines are divided into two parts: one for end users and one for developers

### End Users
1. Download the release from GitHub: [download](https://github.com/daniellamct/Stock-Analysis-Application/releases/download/v1.0.0/Application.zip)

2. Unzip the downloaded folder

3. Run the Application.exe in the folder


### Developers

1. Download the project from GitHub: [repo](https://github.com/daniellamct/Stock-Analysis-Application)

2. Install all the needed libraries by running: pip install -r requirements.txt

3. Run the GUI.py to start the application