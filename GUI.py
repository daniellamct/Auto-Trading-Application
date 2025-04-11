import customtkinter as ctk
from PIL import Image,ImageTk
import yf_check
import threading 
import os

from Graphs_Analysis import GraphsAnalysis as GA
from Company_Profile_Scraping import CompanyProfile as CP
from News_Sentiment_Analysis import NewsSentiment as NS, CardiffnlpInstaller
from Trained_Models import Price_Prediction as PP
from Strategy_Backtesting import backtesting as SB


def show_popup(self, title, message, button="OK", message_type="info"):
    popup = ctk.CTkToplevel(self)
    popup.title(title)
    popup.geometry("400x200")
    
    # Configure grid (1 row, 1 column)
    popup.grid_rowconfigure(0, weight=1)
    popup.grid_columnconfigure(0, weight=1)
    
    # Frame for content
    content = ctk.CTkFrame(popup, corner_radius=10)
    content.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
    
    # Message with icon
    icon = "ⓘ" if message_type == "info" else "⚠" if message_type == "warning" else "❌"
    color = "#1976D2" if message_type == "info" else "#FFA000" if message_type == "warning" else "#D32F2F"
    
    ctk.CTkLabel(content, 
                text=f"{icon}  {title}",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=color).pack(pady=(15, 5), padx=20)
    
    ctk.CTkLabel(content, 
                text=message,
                wraplength=350).pack(pady=5, padx=20)
    if(button!=""):
        ctk.CTkButton(content, 
                    text="OK",
                    command=popup.destroy).pack(pady=15)
    
    popup.grab_set()  # Make it modal



class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # app init
        CardiffnlpInstaller.model_install() # Install models

        folder_name = 'temp' # Create empty folder for temporary storage
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"'{folder_name}' folder created.")
        else:
            print(f"'{folder_name}' folder already exists.")

        # Configure window
        self.title("Stock Analysis Application")
        self.geometry("1400x800")
        self.resizable(1, 1)
        
        ctk.set_default_color_theme("blue")  # Set blue theme


        # Configure grid layout (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1 )
        
        # Create blue navigation frame on the left
        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#1a237e")
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)
        
        # Navigation label
        self.navigation_label = ctk.CTkLabel(self.navigation_frame, 
                                           text="  Menu  ",
                                           font=ctk.CTkFont(size=15, weight="bold"))
        self.navigation_label.grid(row=0, column=0, padx=20, pady=20)



        # Blue navigation buttons with hover effects
        button_style = {
            "fg_color": "#1976D2",  # Blue color
            "hover_color": "#0D47A1",  # Darker blue on hover
            "text_color": "white",
            "font": ctk.CTkFont(size=14),
            "corner_radius": 8,
            "width": 180,
            "height": 40
        }
        
        self.title_button = ctk.CTkButton(self.navigation_frame, text="Introduction Page",
                                        command=self.title_button_event, **button_style)
        self.title_button.grid(row=1, column=0, padx=20, pady=10)
        
        self.graphs_button = ctk.CTkButton(self.navigation_frame, text="Graphs Analysis",
                                         command=self.graphs_button_event, **button_style)
        self.graphs_button.grid(row=2, column=0, padx=20, pady=10)
        
        self.company_button = ctk.CTkButton(self.navigation_frame, text="Company Profile",
                                          command=self.company_button_event, **button_style)
        self.company_button.grid(row=3, column=0, padx=20, pady=10)

        self.news_button = ctk.CTkButton(self.navigation_frame, text="News Analysis",
                                          command=self.news_button_event, **button_style)
        self.news_button.grid(row=4, column=0, padx=20, pady=10)
        
        self.price_button = ctk.CTkButton(self.navigation_frame, text="Price Prediction",
                                          command=self.price_button_event, **button_style)
        self.price_button.grid(row=5, column=0, padx=20, pady=10)
        
        self.sb_button = ctk.CTkButton(self.navigation_frame, text="Strategy Backtesting",
                                          command=self.sb_button_event, **button_style)
        self.sb_button.grid(row=6, column=0, padx=20, pady=10)
        

        # Create container frame for pages
        self.container = ctk.CTkFrame(self, fg_color="transparent" )
        self.container.grid(row=0, column=1, sticky="n" )
        

        # Dictionary to hold frames/pages
        self.frames = {}
        
        # Create all pages
        for Page in (TitlePage, GraphsAnalysisPage, CompanyProfilePage ,NewsAnalysisPage, PricePredictionPage ,StrategyBacktestingPage):
            frame = Page(self.container, self)
            self.frames[Page] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Show the title page initially
        self.show_frame(TitlePage)
        self.select_button(self.title_button)
    
    def select_button(self, button):
        # Reset all buttons
        for btn in [self.title_button, self.graphs_button, self.company_button , self.news_button ,self.price_button, self.sb_button]:
            btn.configure(fg_color="#1976D2")
        
        # Highlight selected button with a darker blue
        button.configure(fg_color="#0D47A1")
    
    def title_button_event(self):
        self.show_frame(TitlePage)
        self.select_button(self.title_button)
    
    def graphs_button_event(self):
        self.show_frame(GraphsAnalysisPage)
        self.select_button(self.graphs_button)
    
    def company_button_event(self):
        self.show_frame(CompanyProfilePage)
        self.select_button(self.company_button)

    def news_button_event(self):
        self.show_frame(NewsAnalysisPage)
        self.select_button(self.news_button)

    def price_button_event(self):
        self.show_frame(PricePredictionPage)
        self.select_button(self.price_button)

    def sb_button_event(self):
        self.show_frame(StrategyBacktestingPage)
        self.select_button(self.sb_button)

    def change_appearance_mode_event(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()



class TitlePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Title label
        title_label = ctk.CTkLabel(self, text="Welcome to the Application", 
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))
        

        # Description
        desc_label = ctk.CTkLabel(self, 
                                text="This application has 6 pages:\n\n\n" \
                                "1. Introduction Page (A brief introduction on the pages)\n\n" \
                                "2. Graphs Analysis (A dashboard to help users analyze the stock)\n\n" \
                                "3. Company Profile (A function to help users understand the company background)\n\n" \
                                "4. News Analysis (A function to fetch latest news about a stock on the internet)\n\n" \
                                "5. Price Prediction (Two machine learning algorithms for price prediction)\n\n" \
                                "6. Strategy Backtesting (A function to perform backtesting on the Crossover Strategy)" ,\
                                font=ctk.CTkFont(size=16))
        desc_label.pack(pady=0)



class GraphsAnalysisPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Configure grid layout (2 rows - inputs and results)
        self.grid_rowconfigure(0, weight=0)  # Input section
        self.grid_rowconfigure(1, weight=1)  # Results section
        self.grid_columnconfigure(0, weight=1)

        
        # Create input container frame (top section)
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        
        # Title Label at top center
        title_label = ctk.CTkLabel(input_frame, 
                                 text="Graphs Analysis",
                                 font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Stock Analysis Input Frame
        stock_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        stock_frame.pack(fill="x", pady=0)
        
        # Analyse Stock input
        ctk.CTkLabel(stock_frame, text="Analyse Stock:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.stock_entry = ctk.CTkEntry(stock_frame, 
                                     placeholder_text="Enter stock symbol...",
                                     height=35)
        self.stock_entry.pack(fill="x", pady=(0, 0))
        
        # Date Range Frame
        date_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        date_frame.pack(fill="x", pady=0)
        
        # Start Date Frame
        start_date_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        start_date_frame.pack(side="left", expand=True, fill="x", padx=0)
        
        ctk.CTkLabel(start_date_frame, text="Start Date:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        # Start Date inputs (Day, Month, Year)
        date_input_frame = ctk.CTkFrame(start_date_frame, fg_color="transparent")
        date_input_frame.pack(fill="x", pady=0)
        
        self.start_day = ctk.CTkEntry(date_input_frame, placeholder_text="DD", width=80, height=35)
        self.start_month = ctk.CTkEntry(date_input_frame, placeholder_text="MM", width=80, height=35)
        self.start_year = ctk.CTkEntry(date_input_frame, placeholder_text="YYYY", width=100, height=35)
        
        self.start_day.pack(side="left", padx=5)
        self.start_month.pack(side="left", padx=5)
        self.start_year.pack(side="left", padx=5)
        
        # End Date Frame
        end_date_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        end_date_frame.pack(side="right", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(end_date_frame, text="End Date:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        
        # End Date inputs (Day, Month, Year)
        date_input_frame = ctk.CTkFrame(end_date_frame, fg_color="transparent")
        date_input_frame.pack(fill="x", pady=0)
        
        self.end_day = ctk.CTkEntry(date_input_frame, placeholder_text="DD", width=80, height=35)
        self.end_month = ctk.CTkEntry(date_input_frame, placeholder_text="MM", width=80, height=35)
        self.end_year = ctk.CTkEntry(date_input_frame, placeholder_text="YYYY", width=100, height=35)
        
        self.end_day.pack(side="left", padx=5)
        self.end_month.pack(side="left", padx=5)
        self.end_year.pack(side="left", padx=5)
        
        # Moving Averages Frame
        ma_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        ma_frame.pack(fill="x", pady=0)
        
        ctk.CTkLabel(ma_frame, text="Moving Averages:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        
        # MA Checkboxes in 2 rows
        ma_values = ["10MA", "20MA", "50MA", "100MA", "150MA", "200MA"]
        self.ma_vars = {}
        
        # First row
        row1_frame = ctk.CTkFrame(ma_frame, fg_color="transparent")
        row1_frame.pack(fill="x", pady=2)
        
        for ma in ma_values[:3]:
            var = ctk.BooleanVar()
            cb = ctk.CTkCheckBox(row1_frame, text=ma, variable=var)
            cb.pack(side="left", padx=15)
            self.ma_vars[ma] = var
            self.boxes=[]
        
        # Second row
        row2_frame = ctk.CTkFrame(ma_frame, fg_color="transparent")
        row2_frame.pack(fill="x", pady=2)
        
        for ma in ma_values[3:]:
            var = ctk.BooleanVar()
            cb = ctk.CTkCheckBox(row2_frame, text=ma, variable=var)
            cb.pack(side="left", padx=15)
            self.ma_vars[ma] = var
            self.boxes=[]
        
        
        # Analyse Button
        analyse_button = ctk.CTkButton(input_frame, 
                                    text="Analyse",
                                    command=self.show_results,
                                    fg_color="#1976D2",
                                    hover_color="#0D47A1",
                                    font=ctk.CTkFont(weight="bold"),
                                    height=40)
        analyse_button.pack(pady=20)
        
        # Results container (bottom section that will expand)
        self.results_frame = ctk.CTkScrollableFrame(self, fg_color="transparent", width=800, height=400)
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.results_frame.grid_columnconfigure(0, weight=1)
   
    def show_results(self):
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        check = yf_check.check_valid(self.start_year.get(),self.start_month.get(),self.start_day.get(),
                    self.end_year.get(),self.end_month.get(),self.end_day.get(),self.stock_entry.get())


        if check:   

            vresult=GA.fun_GA(self.stock_entry.get(),self.start_year.get(),self.start_month.get(),self.start_day.get(),
                    self.end_year.get(),self.end_month.get(),self.end_day.get(),
                    self.ma_vars["10MA"].get(),self.ma_vars["20MA"].get(),self.ma_vars["50MA"].get(),self.ma_vars["100MA"].get(),self.ma_vars["150MA"].get(),self.ma_vars["200MA"].get()
                             
                    )

            if len(vresult)>0:
         
                # Create 3 graph/analysis pairs

                for i in range(3):
                    # Main container for each graph-analysis pair
                    pair_frame = ctk.CTkFrame(self.results_frame, fg_color="transparent")
                    pair_frame.pack(fill="x", pady=(20 if i == 0 else 15, 15))
                    pair_frame.grid_columnconfigure(0, weight=3)  # Graph takes 3/4 space
                    pair_frame.grid_columnconfigure(1, weight=1)  # Analysis takes 1/4 space
                    

                    if i==0:
                        vtitle="Closing Price (Line Chart)"
                    elif i ==1:
                        vtitle="Volume (Line Chart)"
                    elif i==2:
                        vtitle="Daily Price Movement (Violin Plot)" 

                    ctk.CTkLabel(pair_frame,             
                                text=vtitle,
                                font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
                    
                    # Graph placeholder (left side)
                    image_path="temp/ga_"+str(i+1)+".png"

                    image = ctk.CTkImage(light_image=Image.open(image_path),
                                        dark_image=Image.open(image_path),  # Same image for both modes
                                        size=(400, 300))  # Set display size (width, height)
                    graph_placeholder = ctk.CTkLabel(pair_frame,
                                                image=image,
                                                corner_radius=10,text="",
                                                height=300)
                    graph_placeholder.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

                    # Analysis text box (right side)
                    analysis_box = ctk.CTkTextbox(pair_frame,
                                                height=300,
                                                corner_radius=10,
                                                activate_scrollbars=True)
                    analysis_box.insert("0.0", vresult[i])
                    analysis_box.configure(state="disabled")
                    analysis_box.grid(row=1, column=1, sticky="nsew")

            else:
                show_popup(self, "Invalid Input", "Please try again")

        else:
            show_popup(self, "Invalid Input", "Please try again")


    def image_window():
             pass



class CompanyProfilePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Configure grid layout (2 rows - input and results)
        self.grid_rowconfigure(0, weight=0)  # Input section
        self.grid_rowconfigure(1, weight=1)  # Results section
        self.grid_columnconfigure(0, weight=1)
        
        # Create input container frame (top section)
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Title Label
        title_label = ctk.CTkLabel(input_frame, text="Company Profile Scraping", 
                                    font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Stock input frame
        stock_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        stock_frame.pack(fill="x", pady=5)
        
        # Stock input and button
        ctk.CTkLabel(stock_frame, text="Stock:", 
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        
        self.stock_entry = ctk.CTkEntry(stock_frame, 
                                      placeholder_text="Enter stock symbol...",
                                      height=35,
                                      width=200)
        self.stock_entry.pack(side="left", padx=(0, 10))
        
        analyze_button = ctk.CTkButton(stock_frame, 
                                     text="Analyze",
                                     command=self.show_profile,
                                     fg_color="#1976D2",
                                     hover_color="#0D47A1",
                                     font=ctk.CTkFont(weight="bold"),
                                     height=35,
                                     width=100)
        analyze_button.pack(side="left")
        
        # Results container (bottom section that will expand)
        self.results_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.results_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.results_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize profile fields dictionary
        self.profile_fields = {}
    
    def show_profile(self):
      # Parameter validation
        if yf_check.check_stock(self.stock_entry.get()):
            self.result = CP.fun_CP(self.stock_entry.get())
        else:
            self.result = []

        if len(self.result)>0:
            
            # Clear previous results
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            # Create profile fields container
            profile_container = ctk.CTkFrame(self.results_frame, fg_color="transparent")
            profile_container.pack(fill="x", pady=10)
            
            # Define all profile fields with their labels
            fields = [
                ("Company", "The official company name"),
                ("Sector", "The business sector the company operates in"),
                ("Industry", "The specific industry within the sector"),
                ("Location", "Headquarters or primary business location"),
                ("Phone", "Contact phone number"),
                ("URL", "Company website URL"),
                ("Description", "Detailed company description and overview")
            ]
            
            # Create labeled text boxes for each field
            for i, (field_name, placeholder) in enumerate(fields):
                # Field label
                ctk.CTkLabel(profile_container, 
                            text=f"{field_name}:",
                            font=ctk.CTkFont(weight="bold")).grid(row=i, column=0, 
                                                                sticky="w", 
                                                                padx=(0, 10), 
                                                                pady=5)
                
                # Create and store the text box
                text_box = ctk.CTkTextbox(profile_container,
                                        height=40 if field_name != "Description" else 240,
                                        corner_radius=8,
                                        activate_scrollbars=field_name == "Description")
                text_box.insert("0.0", placeholder)
                text_box.configure(state="normal")
                text_box.grid(row=i, column=1, sticky="ew", pady=5)
                
                # Store reference to the text box
                self.profile_fields[field_name.lower()] = text_box
            
            # Configure column weights
            profile_container.grid_columnconfigure(0, weight=0)
            profile_container.grid_columnconfigure(1, weight=1)
            
            # Add some sample data
            self.populate_sample_data()
        else:
            show_popup(self, "Invalid Input", "Please try again")
    
    def populate_sample_data(self):
        sample_data = {
            "company": self.result[0],
            "sector": self.result[1],
            "industry": self.result[2],
            "location": self.result[3],
            "phone": self.result[4],
            "url": self.result[5],
            "description": self.result[6]
        }
        
        for field, value in sample_data.items():
            text_box = self.profile_fields[field]
            text_box.configure(state="normal")
            text_box.delete("0.0", "end")
            text_box.insert("0.0", value)
            text_box.configure(state="disabled")



class NewsAnalysisPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Configure grid layout (3 rows - input, news list, and results)
        self.grid_rowconfigure(0, weight=0)  # Input section
        self.grid_rowconfigure(1, weight=1)  # News list section
        self.grid_rowconfigure(2, weight=1)  # Results section
        self.grid_columnconfigure(0, weight=1)
        
        # Create input container frame (top section)
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Title Label
        title_label = ctk.CTkLabel(input_frame, text="News Analysis", 
                                    font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Stock input frame
        stock_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        stock_frame.pack(fill="x", pady=5)
        
        # Stock input and button
        ctk.CTkLabel(stock_frame, text="Stock:", 
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        
        self.stock_entry = ctk.CTkEntry(stock_frame, 
                                      placeholder_text="Enter stock symbol...",
                                      height=35,
                                      width=200)
        self.stock_entry.pack(side="left", padx=(0, 10))
        
        analyze_button = ctk.CTkButton(stock_frame, 
                                     text="Analyze",
                                     command=self.start_show_news,
                                     fg_color="#1976D2",
                                     hover_color="#0D47A1",
                                     font=ctk.CTkFont(weight="bold"),
                                     height=35,
                                     width=100)
        analyze_button.pack(side="left")
        
        ctk.CTkLabel(stock_frame, text="(It may takes around 30 sec to load)", 
                    font=ctk.CTkFont()).pack(side="left", padx=(10, 10))
        
        # Click the news label
        title_label = ctk.CTkLabel(input_frame, text="Note: Click on the news titles for more information. ", 
                                    font=ctk.CTkFont())
        title_label.pack(side="left", pady=(0, 10))

        # News list scrollable frame (middle section)
        self.news_list_scrollable = ctk.CTkScrollableFrame(self, 
                                                         fg_color="transparent",
                                                         height=150)
        self.news_list_scrollable.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        
        # Results scrollable frame (bottom section)
        self.results_scrollable = ctk.CTkScrollableFrame(self, 
                                                       fg_color="transparent")
        self.results_scrollable.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Initialize news items
        self.news_items = []
        self.current_news_index = None

    def show_news(self):
        # Clear previous results
        for widget in self.news_list_scrollable.winfo_children():
            widget.destroy()
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()
        
        # Parameter validation
        check = yf_check.check_stock(self.stock_entry.get())
        if check:
            result = NS.fun_news(self.stock_entry.get())
        else:
            result = []

        if len(result)>0 and len(result[0])>0:
            # Sample news data 
            sample_news = []
            for v1 in range(len(result[0])):
                sample_news.append({
                            "title": result[0][v1],
                            "textblob": result[1][v1],
                            "cardiffnlp": result[2][v1],
                            "description": result[3][v1],
                            "d100": result[5][v1],
                            "d500": result[4][v1]            
                            }
                            )

            # Create news items in the scrollable list
            self.news_items = []
            for i, news in enumerate(sample_news):
                news_item = self.create_news_item(news, i)
                self.news_items.append(news_item)
                news_item.pack(fill="x", pady=5, padx=5)
        
        else:
            show_popup(self, "Invalid Input", "Please try again")
            
    def start_show_news(self):
        # Create a new thread for long running function 
        news_thread = threading.Thread(target=self.show_news)
        news_thread.start()

    def create_news_item(self, news_data, index):
        # Create frame for each news item
        item_frame = ctk.CTkFrame(self.news_list_scrollable, fg_color="gray20", corner_radius=8)
        
        # Title (clickable)
        title_label = ctk.CTkLabel(item_frame,
                                 text=news_data["title"],
                                 font=ctk.CTkFont(weight="bold"),
                                 cursor="hand2")
        title_label.bind("<Button-1>", lambda e, idx=index: self.show_news_detail(idx))
        title_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        # Ratings frame
        ratings_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        ratings_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # TextBlob rating
        ctk.CTkLabel(ratings_frame,
                    text=f"TextBlob: {news_data['textblob']}",
                    font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        
        # CardiffNLP rating
        ctk.CTkLabel(ratings_frame,
                    text=f"CardiffNLP: {news_data['cardiffnlp']}",
                    font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        
        # Store news data with the frame
        item_frame.news_data = news_data
        return item_frame
    
    def show_news_detail(self, index):
        # Clear previous detail view
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()
        
        # Store current news index
        self.current_news_index = index
        
        # Get news data
        news_data = self.news_items[index].news_data
        
        # Summarization buttons
        summarization_frame = ctk.CTkFrame(self.results_scrollable, fg_color="transparent")
        summarization_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(summarization_frame, 
                    text="Summarization:",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(summarization_frame,
                     text="Original",
                     command=lambda: self.update_description(news_data["description"]),
                     width=80).pack(side="left", padx=5)
        
        ctk.CTkButton(summarization_frame,
                     text="500 words",
                     command=lambda: self.update_description(news_data["d500"]),
                     width=80).pack(side="left", padx=5)
        
        ctk.CTkButton(summarization_frame,
                     text="100 words",
                     command=lambda: self.update_description(news_data["d100"]),
                     width=80).pack(side="left", padx=5)
        
        # News title
        ctk.CTkLabel(self.results_scrollable,
                    text=f"Title: {news_data['title']}",
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 10))
        
        # News description
        self.detail_text = ctk.CTkTextbox(self.results_scrollable,
                                        height=300,
                                        corner_radius=8,
                                        wrap="word",
                                        activate_scrollbars=True)
        self.detail_text.insert("1.0", news_data["description"])
        self.detail_text.configure(state="disabled")
        self.detail_text.pack(fill="x", expand=True)
    
    def update_description(self, text):
        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", text)
        self.detail_text.configure(state="disabled")
    
    def summarize_text(self, text, max_words):
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + "... [continued]"



class PricePredictionPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Configure grid layout
        self.grid_rowconfigure(0, weight=0)  # Input section
        self.grid_rowconfigure(1, weight=1)  # Results section
        self.grid_columnconfigure(0, weight=1)
        
        # Create input container frame
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Title Label
        title_label = ctk.CTkLabel(input_frame, text="Price Prediction", 
                                    font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Stock input and button
        ctk.CTkLabel(input_frame, 
                    text="Stock for Price Prediction:",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        
        self.stock_entry = ctk.CTkEntry(input_frame, 
                                      placeholder_text="Enter stock symbol...",
                                      height=35,
                                      width=200)
        self.stock_entry.pack(side="left", padx=(0, 10))
        
        predict_button = ctk.CTkButton(input_frame, 
                                     text="Predict",
                                     command=self.show_prediction,
                                     fg_color="#1976D2",
                                     hover_color="#0D47A1",
                                     font=ctk.CTkFont(weight="bold"),
                                     height=35,
                                     width=100)
        predict_button.pack(side="left")
        
        # Create scrollable results container
        self.results_scrollable = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.results_scrollable.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Initialize prediction display references
        self.ann_graph = None
        self.lstm_graph = None
    
    def show_prediction(self):
        # Parameter validation
        check = yf_check.check_stock(self.stock_entry.get())
        if check:
            self.result=PP.fun_PP(self.stock_entry.get())
        else:
            self.result = []

        if(len(self.result)>0):
            # Clear previous results
            for widget in self.results_scrollable.winfo_children():
                widget.destroy()
            
            # Create ANN section
            self.create_model_section("ANN")
            
            # Add separator
            ctk.CTkLabel(self.results_scrollable, 
                        text="────────────────────────────",
                        font=ctk.CTkFont(size=14)).pack(pady=20)
            
            # Create LSTM section
            self.create_model_section("LSTM")

        else:
            show_popup(self, "Invalid Input", "Please try again")

            
    
    def create_model_section(self, model_name):
        # Model label
        ctk.CTkLabel(self.results_scrollable,
                    text=f"{model_name}:",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(10, 5))
        if model_name == "ANN":
            ctk.CTkLabel(self.results_scrollable,
            text="(a prediction method that evaluates each sample individually to make predictions)",
            font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5))
        elif model_name == "LSTM":
            ctk.CTkLabel(self.results_scrollable,
            text="(a prediction method that considers historical data to improve its forecasts)",
            font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5)) 
            
        # Prediction graph 
        if(model_name=="ANN"):
            image_path="temp/PP0.png" 
            v1=1
            v2=0
            v3=2
        if(model_name=="LSTM"):
            image_path="temp/PP1.png"
            v1=4
            v2=3
            v3=5

        image = ctk.CTkImage(light_image=Image.open(image_path),
                                dark_image=Image.open(image_path),  # Same image for both modes
                                size=(700, 300))  # Set display size (width, height)

        graph_frame = ctk.CTkFrame(self.results_scrollable, fg_color="transparent")
        graph_frame.pack(fill="x", pady=5)
        
        graph_label = ctk.CTkLabel(graph_frame,
                                 text=f"{model_name} Prediction Graph",
                                 fg_color="gray20",
                                 corner_radius=10,
                                 height=300, image=image)
        graph_label.pack(fill="x")

        # Prediction text box
        metrics_box1 = ctk.CTkTextbox(self.results_scrollable,
                                    height=100,
                                    width=700,
                                    corner_radius=8,
                                    activate_scrollbars=False)
        
        vtext = "\n".join(self.result[v2])
        metrics_box1.insert("1.0", vtext)
        metrics_box1.configure(state="disabled")
        metrics_box1.pack(pady=10)

        # Container for the backtesting section
        backtesting_frame = ctk.CTkFrame(self.results_scrollable, fg_color="transparent")
        backtesting_frame.pack(fill="x", pady=10)
        
        # Backtesting Result label
        ctk.CTkLabel(backtesting_frame,
                    text="Backtesting Result:",
                    font=ctk.CTkFont(weight="bold",size=14)).pack(anchor="w", pady=(0, 10))

        # Container for the 2x2 grid and text box
        results_frame = ctk.CTkFrame(backtesting_frame, fg_color="transparent")
        results_frame.pack(fill="x")
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_columnconfigure(1, weight=1)

        # Confusion matrix (2x2 grid) 
        matrix_frame = ctk.CTkFrame(results_frame, fg_color="transparent")
        matrix_frame.grid(row=0, column=0, sticky="nsew", padx=10)

        # 2x2 grid for confusion matrix
        for i in range(2):
            for j in range(2):
                cell = ctk.CTkFrame(matrix_frame, 
                                fg_color="gray25",
                                corner_radius=0,
                                border_width=1,
                                border_color="gray40")
                cell.grid(row=i, column=j, sticky="nsew", padx=1, pady=1)
                matrix_frame.grid_rowconfigure(i, weight=1)
                matrix_frame.grid_columnconfigure(j, weight=1)
                
                # Add labels to each quadrant
                if i == 0 and j == 0:
                    label = "TP"
                    value = self.result[v1][0]
                elif i == 0 and j == 1:
                    label = "FP"
                    value = self.result[v1][2]
                elif i == 1 and j == 0:
                    label = "FN"
                    value = self.result[v1][3]
                else:
                    label = "TN"
                    value = self.result[v1][1]
                
                ctk.CTkLabel(cell,
                            text=f"{label}\n{value}",
                            font=ctk.CTkFont(size=12, weight="bold")).pack(expand=True)

        # Prediction evaluation text box on right side
        metrics_box2 = ctk.CTkTextbox(results_frame,
                                    height=100,
                                    width=300,
                                    corner_radius=8,
                                    activate_scrollbars=False)
        
        vtext = "\n".join(self.result[v3])
        metrics_box2.insert("1.0", vtext)
        metrics_box2.configure(state="disabled")
        metrics_box2.grid(row=0, column=1, sticky="nsew", padx=10)

        # Store graph reference
        if model_name == "ANN":
            self.ann_graph = graph_label
        else:
            self.lstm_graph = graph_label
    
    def update_prediction(self, stock_symbol):
        
        # Update the stock symbol in the graphs
        if self.ann_graph:
            self.ann_graph.configure(text=f"ANN Prediction Graph for {stock_symbol}")
        if self.lstm_graph:
            self.lstm_graph.configure(text=f"LSTM Prediction Graph for {stock_symbol}")



class StrategyBacktestingPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        
        # Create main container
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title Label
        title_label = ctk.CTkLabel(container, text="Moving Average Crossover Backtesting", 
                                    font=ctk.CTkFont(size=24, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Stock Input Frame
        stock_frame = ctk.CTkFrame(container, fg_color="transparent")
        stock_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(stock_frame, text="Stock:", 
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        
        self.stock_entry = ctk.CTkEntry(stock_frame, 
                                      placeholder_text="e.g. AAPL   e.g. AAPL AMZN GOOG",
                                      height=35)
        self.stock_entry.pack(fill="x", expand=True)
        
        # Date Range Frame
        date_frame = ctk.CTkFrame(container, fg_color="transparent")
        date_frame.pack(fill="x", pady=10)
        
        # Start Date Frame
        start_date_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        start_date_frame.pack(side="left", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(start_date_frame, text="Start Date:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        
        # Start Date inputs
        start_input_frame = ctk.CTkFrame(start_date_frame, fg_color="transparent")
        start_input_frame.pack(fill="x")
        
        self.start_day = ctk.CTkEntry(start_input_frame, placeholder_text="DD", width=50)
        self.start_month = ctk.CTkEntry(start_input_frame, placeholder_text="MM", width=50)
        self.start_year = ctk.CTkEntry(start_input_frame, placeholder_text="YYYY", width=70)
        
        self.start_day.pack(side="left", padx=2)
        self.start_month.pack(side="left", padx=2)
        self.start_year.pack(side="left", padx=2)
        
        # End Date Frame
        end_date_frame = ctk.CTkFrame(date_frame, fg_color="transparent")
        end_date_frame.pack(side="right", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(end_date_frame, text="End Date:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        
        # End Date inputs
        end_input_frame = ctk.CTkFrame(end_date_frame, fg_color="transparent")
        end_input_frame.pack(fill="x")
        
        self.end_day = ctk.CTkEntry(end_input_frame, placeholder_text="DD", width=50)
        self.end_month = ctk.CTkEntry(end_input_frame, placeholder_text="MM", width=50)
        self.end_year = ctk.CTkEntry(end_input_frame, placeholder_text="YYYY", width=70)
        
        self.end_day.pack(side="left", padx=2)
        self.end_month.pack(side="left", padx=2)
        self.end_year.pack(side="left", padx=2)
        
        # MA Lines Frame
        ma_frame = ctk.CTkFrame(container, fg_color="transparent")
        ma_frame.pack(fill="x", pady=15)
        
        # Fast Line input
        fast_frame = ctk.CTkFrame(ma_frame, fg_color="transparent")
        fast_frame.pack(side="left", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(fast_frame, text="Fast Line:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        
        self.fast_line = ctk.CTkEntry(fast_frame, placeholder_text="e.g. 10")
        self.fast_line.pack(fill="x")
        
        # Slow Line input
        slow_frame = ctk.CTkFrame(ma_frame, fg_color="transparent")
        slow_frame.pack(side="right", expand=True, fill="x", padx=5)
        
        ctk.CTkLabel(slow_frame, text="Slow Line:", 
                    font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        
        self.slow_line = ctk.CTkEntry(slow_frame, placeholder_text="e.g. 30")
        self.slow_line.pack(fill="x")
        
        # Analyze Button
        analyze_button = ctk.CTkButton(container, 
                                     text="Analyze",
                                     command=self.run_backtest,
                                     fg_color="#1976D2",
                                     hover_color="#0D47A1",
                                     font=ctk.CTkFont(weight="bold"),
                                     height=40)
        analyze_button.pack(pady=20)
        
        # Create scrollable results container (initially empty)
        self.results_scrollable = ctk.CTkScrollableFrame(container, fg_color="transparent")
        self.results_scrollable.pack(fill="both", expand=True)
        
        # Will be populated after analysis
        self.results_container = None
    
    def run_backtest(self):
        """Function to execute when Analyze button is clicked"""
        # Clear previous results
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()

                
        # Create new results container
        self.results_container = ctk.CTkFrame(self.results_scrollable, fg_color="transparent")
        self.results_container.pack(fill="both", expand=True, pady=10)
            
        # Parameters validation
        stocks = self.stock_entry.get().split()
        flag = True

        for stock in stocks: # Stock validation
            check = yf_check.check_stock(stock)  
            if check == False:
                flag = False
            
        check = yf_check.check_num(self.fast_line.get(), self.slow_line.get()) # Fast and Slow line validation
        if check == False:
            flag = False              
        
        check = yf_check.check_date(self.start_year.get(), self.start_month.get(), self.start_day.get(), 
                                    self.end_year.get(), self.end_month.get(), self.end_day.get()) # Date range validation
        if check == False:
                flag = False 

        if flag:
            vresult = SB.fun_sb(self.stock_entry.get(),
                                self.start_year.get(), self.start_month.get(), self.start_day.get(),
                                self.end_year.get(), self.end_month.get(), self.end_day.get(),
                                self.fast_line.get(), self.slow_line.get())
        else:
            vresult = []

     
        if len(vresult) > 0:
            lidt_stocks = self.stock_entry.get().split()

            # Overall Results Section
            overall_frame = ctk.CTkFrame(self.results_container, fg_color="transparent")
            overall_frame.pack(fill="x", pady=10)
            
            ctk.CTkLabel(overall_frame, 
                        text="Overall:",
                        font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
            
            overall_text = ctk.CTkTextbox(overall_frame,
                                        height=100,
                                        corner_radius=8,
                                        activate_scrollbars=False)
            overall_text.insert("1.0",vresult[len(vresult)-1])#"1.0", 
            overall_text.configure(state="disabled")
            overall_text.pack(fill="x", pady=5)
            

            for i in range(len(vresult)-1):
                
                # Stock-Specific Results Section
                stock_frame = ctk.CTkFrame(self.results_container, fg_color="transparent")
                stock_frame.pack(fill="x", pady=10)
                
                ctk.CTkLabel(stock_frame, 
                            text=f"{lidt_stocks[i]}:",
                            font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
                
                # Image and text box in a horizontal layout
                stock_results_frame = ctk.CTkFrame(stock_frame, fg_color="transparent")
                stock_results_frame.pack(fill="x")
                


                image_path="temp/sb_"+str(i)+".png"

                im_sn = ctk.CTkImage(light_image=Image.open(image_path),
                                    dark_image=Image.open(image_path),  # Same image for both modes
                                    size=(550, 300))  # Set display size (width, height)


                # Image placeholder 
                image_placeholder = ctk.CTkLabel(stock_results_frame,
                                            fg_color="gray20",
                                            corner_radius=10,
                                            width=550,
                                            height=300, image=im_sn          
                                            )
                image_placeholder.pack(side="left", padx=(0, 10))
                
                # Results text box
                results_text = ctk.CTkTextbox(stock_results_frame,
                                            height=250,
                                            width=200,
                                            corner_radius=8,
                                            activate_scrollbars=False)
                results_text.insert("1.0",vresult[i])
                results_text.configure(state="disabled")
                results_text.pack(side="left")

        else:
            show_popup(self, "Invalid Input", "Please try again")                            



if __name__ == "__main__":
    app = App()
    app.mainloop()

