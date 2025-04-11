import requests
from bs4 import BeautifulSoup


def fun_CP(stock):

    try:
        # Web Scraping
        url = f'https://finance.yahoo.com/quote/{stock}/profile/'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        received_content = response.text
        soup = BeautifulSoup(received_content, 'html.parser')


        companyName = ""
        sector = ""
        industry = ""
        location = ""
        phone = ""
        companyURL = ""
        description = ""

        # Company Name extraction
        companyName_area = soup.find('section', class_='container yf-xxbei9 paddingRight')
        if companyName_area:
            cn_area = companyName_area.find('h1', class_='yf-xxbei9')
            if cn_area:
                companyName = cn_area.text

        # Sector and Industry extraction
        stats_area = soup.find('dl', class_='company-stats')
        if stats_area:
            stats = stats_area.find_all('a')
            if stats[1]:
                sector = stats[0].text
                industry = stats[1].text

        # Location extraction
        location_area = soup.find('div', class_='address')
        if location_area:
            loca = location_area.find_all('div')
            if loca[2]:
                location = loca[2].text

        # Phone extraction
        phone_area = soup.find('a', {'data-ylk': 'elm:company;elmt:link;itc:0;sec:qsp-company-overview;subsec:profile;slk:business-phone'})
        if phone_area:
            phone = phone_area.text

        # Company URL extraction
        url_area = soup.find('a', {'data-ylk': 'elm:company;elmt:link;itc:0;sec:qsp-company-overview;subsec:profile;slk:business-url'})
        if url_area:
            companyURL = url_area.text

        # Description extraction
        description_area = soup.find('section', {'data-testid': 'description'})
        if description_area:
            des_area = description_area.find('p')
            if des_area:
                description = des_area.text


        print("\n")
        print(f'Company: {companyName}\n')
        print(f'Sector: {sector}\n')
        print(f'Industry: {industry}\n')
        print(f'Location: {location}\n')
        print(f'Phone: {phone}\n')
        print(f'URL: {companyURL}\n')
        print(f'Description: {description}\n')

        return [companyName,sector,industry,location ,phone,companyURL,description]
    
    except:
        return []

