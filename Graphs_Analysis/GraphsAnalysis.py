import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns


def fun_GA(stock, sy, sm, sd, ey, em, ed, MA10_show, MA20_show, MA50_show, MA100_show, MA150_show, MA200_show):

  try:
    vreturn=[]

    # Download dataset
    start_year = sy
    start_month = sm
    start_day = sd

    end_year = ey
    end_month = em
    end_day = ed

    start_date = datetime(int(start_year), int(start_month), int(start_day))
    end_date = datetime(int(end_year), int(end_month), int(end_day))

    new_start_date = start_date - timedelta(days=400)

    data = yf.download(stock, new_start_date, end_date,progress=False)


    # Feature Calculation
    data['Changed_1day'] = data['Close'] - data['Close'].shift(1)
    data['10_MA'] = data['Close'].rolling(window=10).mean()
    data['20_MA'] = data['Close'].rolling(window=20).mean()
    data['50_MA'] = data['Close'].rolling(window=50).mean()
    data['100_MA'] = data['Close'].rolling(window=100).mean()
    data['150_MA'] = data['Close'].rolling(window=150).mean()
    data['200_MA'] = data['Close'].rolling(window=200).mean()

    data = data[data.index >= start_date]

    volume = data['Volume'].to_numpy().flatten()
    changed_1day = data['Changed_1day'].to_numpy().flatten()
    prices = data['Close'].to_numpy().flatten()
    MA10 = data['10_MA'].to_numpy().flatten()
    MA20 = data['20_MA'].to_numpy().flatten()
    MA50 = data['50_MA'].to_numpy().flatten()
    MA100 = data['100_MA'].to_numpy().flatten()
    MA150 = data['150_MA'].to_numpy().flatten()
    MA200 = data['200_MA'].to_numpy().flatten()


    # Closing Price (Line Chart)
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, data.Close, label="Closing Price", color="b")
    if MA10_show:
        plt.plot(data.index, MA10, label="10 Moving Average", color="r")
    if MA20_show:
        plt.plot(data.index, MA20, label="20 Moving Average", color="c")
    if MA50_show:
        plt.plot(data.index, MA50, label="50 Moving Average", color="m")
    if MA100_show:
        plt.plot(data.index, MA100, label="100 Moving Average", color="brown")
    if MA150_show:
        plt.plot(data.index, MA150, label="150 Moving Average", color="pink")
    if MA200_show:
        plt.plot(data.index, MA200, label="200 Moving Average", color="orange")

    plt.legend()
    plt.title(f'Closing Price (Line Chart) - {stock}')
    plt.xlabel("Days")
    plt.ylabel("Closing Price")
    plt.tight_layout()

    highest_ClosingPrice = f"{np.max(prices):.2f}"
    lowest_ClosingPrice = f"{np.min(prices):.2f}"
    average_ClosingPrice = f"{np.mean(prices):.2f}"
    median_ClosingPrice = f"{np.median(prices):.2f}"

    print("\nClosing Price: ")
    print(f"Highest: {highest_ClosingPrice}")
    print(f"Lowest: {lowest_ClosingPrice}")
    print(f"Average: {average_ClosingPrice}")
    print(f"Median: {median_ClosingPrice}")

    vreturn.append(("Closing Price: ")+"\n"+
        (f"Highest: {highest_ClosingPrice}")+"\n"+
        (f"Lowest: {lowest_ClosingPrice}")+"\n"+
        (f"Average: {average_ClosingPrice}")+"\n"+
        (f"Median: {median_ClosingPrice}"))

    plt.savefig("temp/ga_1.png")


    # Volume (Line Chart)
    plt.figure(figsize=(8, 6))
    plt.plot(data.index, volume, label="Volume", color="b")
    plt.legend()
    plt.title(f'Volume (Line Chart) - {stock}')
    plt.xlabel("Days")
    plt.ylabel("Volume")
    plt.tight_layout()

    highest_Volume = np.max(volume)
    lowest_Volume = np.min(volume)
    average_Volume = f"{np.mean(volume):.2f}"
    median_Volume = f"{np.median(volume):.2f}"

    print("\nVolume: ")
    print(f"Highest: {highest_Volume}")
    print(f"Lowest: {lowest_Volume}")
    print(f"Average: {average_Volume}")
    print(f"Median: {median_Volume}")

    plt.savefig("temp/ga_2.png")

    vreturn.append("Volume: "+"\n"+
        (f"Highest: {highest_Volume}")+"\n"+
        (f"Lowest: {lowest_Volume}")+"\n"+
        (f"Average: {average_Volume}")+"\n"+
        (f"Median: {median_Volume}"))


    # Daily Price Movement (Violin Plot) 
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=changed_1day, color='lightblue', inner='quartile')
    plt.ylabel('Price Movement')
    plt.title(f'Daily Price Movement (Violin Plot) - {stock}')
    plt.grid(axis='y', alpha=0.75)

    highest_PriceMovement = f"{np.max(changed_1day):.2f}"
    lowest_PriceMovement = f"{np.min(changed_1day):.2f}"
    average_PriceMovement = f"{np.mean(changed_1day):.2f}"
    median_PriceMovement = f"{np.median(changed_1day):.2f}"
    average_abs_PriceMovement = f"{np.mean(np.abs(changed_1day)):.2f}"
    median_abs_PriceMovement = f"{np.median(np.abs(changed_1day)):.2f}"

    print("\nDaily Price Movement: ")
    print(f"Highest: {highest_PriceMovement}")
    print(f"Lowest: {lowest_PriceMovement}")
    print(f"Average: {average_PriceMovement}")
    print(f"Median: {median_PriceMovement}")
    print(f"Average Absolute Movement: {average_abs_PriceMovement}")
    print(f"Median Absolute Movement: {median_abs_PriceMovement}")
    
    plt.savefig("temp/ga_3.png")

    vreturn.append("Daily Price Movement: "+"\n"+
        (f"Highest: {highest_PriceMovement}")+"\n"+
        (f"Lowest: {lowest_PriceMovement}")+"\n"+
        (f"Average: {average_PriceMovement}")+"\n"+
        (f"Median: {median_PriceMovement}")+"\n"+
        (f"Average Absolute Movement: {average_abs_PriceMovement}")+"\n"+
        (f"Median Absolute Movement: {median_abs_PriceMovement}"))

    return vreturn
  except:
    return []
  


