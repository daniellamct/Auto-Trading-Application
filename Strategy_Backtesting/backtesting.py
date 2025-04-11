import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fun_sb(input_stock, start_year, start_month, start_day, end_year, end_month, end_day, fast, slow):

        try:
            vreturn=[]
            stocks = input_stock.split()
     
            # Preprocessing
            start_date = datetime(int(start_year), int(start_month), int(start_day))
            end_date = datetime(int(end_year), int(end_month), int(end_day))

            slow=int(slow)
            fast=int(fast)

            new_start_date = start_date - timedelta(days=slow*2)
            new_start_date = new_start_date.strftime('%Y-%m-%d')

            fast_line = str(fast)+'MA'
            slow_line = str(slow)+'MA'

            data = {}
            total_t = 0
            total_win_t = 0
            total_earn = 0

            print(f'Start Analyzing: From {start_date} to {end_date}')
            

            v1=0

            for stock in stocks:
                # Download the dataset
                data[stock] = yf.download(stock, start=new_start_date, end=end_date,progress=False)

                data[stock][fast_line] = data[stock]['Close'].rolling(window=fast).mean()
                data[stock][slow_line] = data[stock]['Close'].rolling(window=slow).mean()

                data[stock] = data[stock][data[stock].index >= start_date]

                active = False
                buy_points = []
                sell_points = []
                local_t = 0
                local_win_t = 0
                local_earn = 0

                print(".............................")
                print(stock + ":")
                for i in range(1, data[stock].shape[0]):
                        if (data[stock].iloc[i][fast_line].item() >= data[stock].iloc[i][slow_line].item()) and (data[stock].iloc[i-1][fast_line].item() < data[stock].iloc[i-1][slow_line].item()) and not active:
                                # Buy 
                                active = True
                                price = data[stock].iloc[i]['Close'].item()
                                cur = price
                                buy_points.append((data[stock].index[i], price))
                                print(f"Buy: {price:.2f}")

                        elif (data[stock].iloc[i][fast_line].item() < data[stock].iloc[i][slow_line].item()) and (data[stock].iloc[i-1][fast_line].item() >= data[stock].iloc[i-1][slow_line].item()) and active:
                                # Sell
                                active = False
                                price = data[stock].iloc[i]['Close'].item()
                                total_earn = total_earn - cur + price
                                local_earn = local_earn -cur + price
                                total_t += 1
                                local_t += 1
                                if cur < price:
                                    total_win_t += 1
                                    local_win_t += 1
                                
                                sell_points.append((data[stock].index[i], price))
                                print(f"Sell: {price:.2f}")

                if local_t > 0:
                    local_acc = f'{local_win_t}/{local_t} ({(local_win_t/local_t)*100:.2f}%)'
                else:
                    local_acc = f'none'

                # Result for one stock
                print(f"\nResult: ")
                print(f"Earned: {local_earn:.2f} USD")
                print(f"Total Completed Trades: {local_t}")
                print("Accuracy: " + local_acc)

                vreturn.append(("Result: ")+"\n"+
                            (f"Earned: {local_earn:.2f} USD")+"\n"+
                            (f"Total Completed Trades: {local_t}") +"\n"+
                            ("Accuracy: " + local_acc))

                # Plot the graph
                plt.figure(figsize=(11, 6))
                plt.plot(data[stock]['Close'], label='Close Price', color='blue', alpha=0.5)
                plt.plot(data[stock][fast_line], label=str(fast)+'-day MA', color='orange')
                plt.plot(data[stock][slow_line], label=str(slow)+'-day MA', color='purple')

                # Plot buy and sell points
                if local_t > 0:
                    buy_x, buy_y = zip(*buy_points)
                    sell_x, sell_y = zip(*sell_points)
                    plt.scatter(buy_x, buy_y, color='green', marker='x', s=300, alpha=1, label='Buy Signal', linewidths=3, zorder=5)
                    plt.scatter(sell_x, sell_y, color='red', marker='x', s=300, alpha=1, label='Sell Signal', linewidths=3, zorder=5)
                elif active:
                    buy_x, buy_y = zip(*buy_points)
                    plt.scatter(buy_x, buy_y, color='green', marker='x', s=300, alpha=1, label='Buy Signal', linewidths=3, zorder=5)

                plt.title(f'Moving Average Crossover Backtesting Analysis for stock - {stock}')
                plt.xlabel('Date')
                plt.ylabel('Closing Price')
                plt.legend(markerscale=0.4)
                plt.grid(True)

                plt.savefig("temp/sb_"+str(v1)+".png")
                v1+=1

            if total_t > 0:
                acc = f'{total_win_t}/{total_t} ({(total_win_t/total_t)*100:.2f}%)'
            else:
                acc = f'none'

            # Overall Result
            print(".............................")
            print("Overall: ")
            print(f"Earned: {total_earn:.2f} USD")
            print(f"Total Completed Trades: {total_t}")
            print("Accuracy: " + acc)


            vreturn.append(("Overall: ")+"\n"+
                        (f"Earned: {total_earn:.2f} USD")+"\n"+
                        (f"Total Completed Trades: {total_t}")+"\n"+
                        ("Accuracy: " + acc))

            return vreturn
           
        except:
             return []
