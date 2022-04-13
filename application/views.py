import datetime
import json
from turtle import distance
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from goose3 import Goose
from bs4 import BeautifulSoup #Import stuff
import requests
import os
from pathlib import Path
import shutil
from newspaper import Article
import nltk
from langdetect import detect
from deep_translator import GoogleTranslator
from pytimeextractor import ExtractionService, PySettingsBuilder
import pandas as pd
from datetime import date
import string    
import inquirer
import spacy
import pandas as pd
import geopy 
import math as Math
import inquirer
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from googlesearch import search 
import requests
from bs4 import BeautifulSoup
import pandas as pd
import inquirer
import itertools
import os
from selenium import webdriver
import requests
import inquirer
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from os import listdir
from os.path import join, isfile


def index(request):
    return render(request,'home/index.html')

def analysis(request):
    if request.method == 'POST':
        url = request.POST.get('url',None)
        author = []
        score = 0
        analysis.text, analysis.auth, analysis.new_score, analysis.title, analysis.links = url_analysis(url,score)
        request.session['text_data'] = analysis.text
        request.session['title'] = analysis.title
    return HttpResponseRedirect('who')

def who(request):
    analysis(request)
    auth = analysis.auth
    author = []
    if(len(auth) > 0):
        for x in auth:
            name = {'name' : x}
    else:
        name = {'name' : 'No Author Found'}
    author.append(name)
    return render(request,'home/who.html',{'author' : author})

def when(request):
    if request.method == 'POST':
        author = request.POST.get('author',None)
        if author == "No":
            who_score = 0
        else:
            who_score =1
    # analysis(request)
    request.session['who_score'] = who_score
    text = request.session['text_data']
    lang = detect(text)
    if lang == 'en':
        text = text
    else:
        to_translate = text
        translated = GoogleTranslator(source='auto', target='en').translate(to_translate)
        text = translated
    date_list = when_analysis(text)
    return render(request,'home/when.html',{'date_list':date_list})
    # ?return render(request,'home/index.html')

def what(request):
    topic = ['No topic can be identified','News item','Scientific publication','Dogma','Maximum phrase','Political propaganda','Commercial propaganda','Satire']  
    return render(request,'home/what.html',{'topic':topic})

def where(request):
    text = request.session['text_data']
    location,where.df = where_analysis(request,text)
    return render(request,'home/where.html',{'location':location})

def why(request):
    choice = ['YES','NO']
    return render(request,'home/why.html',{'choice':choice})

def how(request):
    if request.method == 'POST':
        choice = request.POST.get('choice', None)
        if choice == 'YES':
            score = 0
        else:
            score = 1
        request.session['why_score'] = score
    
    # analysis(request)
    title = request.session['title'] 
    rep, title_score = count_upper(title)
    request.session['how1'] = title_score
    
    search=searchG(title)
    print(title_score)
    return render(request,'home/how.html',{'search':search})

def final_analysis(request):
    if request.method == 'POST':
        choice = request.POST.get('choice',None)
        if choice ==  "No Link identified || No Title Identified": 
            h2_score = 0
        else:
            h2_score = 1
        request.session['how2'] = h2_score
    img_item, not_approved = ImageAnalize()
    image = []
    quest = []
    if(img_item != None and not_approved != None ):
        image_score = 1; 
    else:
        image_score = 1;
    request.session['how3'] = image_score

    if img_item != None:
        for itm in img_item:
            for k, v in itm.items():
                image.append(k)
                quest.append(v)
    else:
        return HttpResponseRedirect('result')
    # print(not_approved)

    context=zip(image,quest)
    return render(request,'home/fanalysis1.html',{'context':context, 'not_approved':not_approved})

def result(request):
    # if request.method=='POST':
        # who(request)
        # s1 = who.who_score
        # when(request)
        # s2= when_score.score
        # what(request)
    s1 = request.session['who_score']
    s2 = request.session['when_score']
    s3 = request.session['what_score']
    s4 = request.session['where_score']
    s5 = request.session['why_score']
    s6 = request.session['how1']
    s7 = request.session['how2']
    s8 = request.session['how3']
    print(s1, s2, s3, s4, s5, s6, s7, s8)
    total_score = s1+s2+s3+s4+s5+s6+s7+s8 
    print("totalscore", total_score)
    accuracy = []
    percentage = []
    per = (total_score * 100) / 8.7
    percentage.append(round(per,2))
    if total_score > 5.4:
        accuracy.append("Real News")
    else: 
        accuracy.append("Fake News")
    return render(request,'home/result.html',{'accuracy':accuracy,'percentage':percentage})

def url_analysis(url,score):
    g = Goose()
    article = g.extract(url=url)
    text=article.cleaned_text
    title=(article.title)
#    print(article.authors)

    articleN = Article(url)
    articleN.download() #Downloads the linkâ€™s HTML content
    articleN.parse() #Parse the article
    nltk.download('punkt')#1 time download of the sentence tokenizer
    articleN.nlp()#  Keyword extraction wrapper
    
    r  = requests.get(url) #Download website source
    data = r.text  #Get the website source as text
    
    # 'Scaricamento immagini'
    soup = BeautifulSoup(data, 'html.parser') 
    links = []
    for link in soup.find_all('img'): 
        imgSrc = link.get('src')   
        links.append(imgSrc)    

    if not os.path.isdir("media/articleImage/"):
        os.mkdir("media/articleImage/")
    else:
        dir_path=Path("media/articleImage/")
        shutil.rmtree(dir_path)
        os.mkdir("media/articleImage/")
    
    countImage=1
    for el in links:
        if el and (el.startswith('http')):
            response = requests.get(el)
            fi=open("media/articleImage/image"+str(countImage)+".jpg", "wb")
            fi.write(response.content)
            fi.close()
            countImage+=1
        else:
            continue

    if (article.authors == []):
        auth=article.authors
        score=0
    else:
        auth= article.authors
        score=1
    return(text, auth, score, title, links)

def what_score(request):
    if request.method == 'POST':
        topic = request.POST.get('topic',None)
   
        if (topic == 'No topic can be identified'):
            score=0
        elif (topic == 'News item' or topic == 'Scientific publication'):
            score=1
    else:
        score=0.5
    request.session['what_score'] = score
    # print(score, what.score)    
    # return(score)
    return HttpResponseRedirect('where')

def when_analysis(text):
    settings = (PySettingsBuilder()
    .excludeRules("durationRule")
    .excludeRules("repeatedRule")
    .excludeRules("timeIntervalRule")
    .build())

    textElim=''.join(i for i in text if i in string.printable)    #Eliminazione caratteri speciali 
    
    result = ExtractionService.extract(textElim, settings);
    NoTime={'temporalExpression': 'No Time Identified', 'fromPosition': 0, 'toPosition': 0, 'classOfRuleType': 'No Time', 'temporal': [{'rule': 'holidaysRule', 'group': 'DateGroup', 'duration': None, 'durationInterval': None, 'set': None, 'type': 'DATE', 'startDate': {'time': {'hours': 0, 'minutes': 0, 'seconds': 0, 'timezoneName': None, 'timezoneOffset': 0}, 'date': {'year': 0000, 'month': 00, 'day': 00, 'dayOfWeek': None, 'weekOfMonth': None}, 'relative': True}, 'endDate': {'time': {'hours': 00, 'minutes': 0, 'seconds': 00, 'timezoneName': None, 'timezoneOffset': 0}, 'date': {'year': 0000, 'month': 00, 'day': 00, 'dayOfWeek': None, 'weekOfMonth': None}, 'relative': True}}], 'confidence': 0.99, 'locale': 'en_US', 'rule': {'rule': "((halloween)|(christmas eve)|(christmas day)|(christmas)|(new year's day)|(new year day)|(New Year s' Eve)|(New Year's Eve)|(new year)|(independence day)|(thanksgiving day)|(thanksgiving)|(Veterans Day)|(Columbus Day)|(Labor Day)|(Memorial Day)|(Washington's Birthday)|(Martin Luther King, Jr. Day)|(Martin Luther King Day)|(Inauguration Day)|((st[.]?|saint)[\\s]*(valentine|valentine's|valentines)[\\s]*(day)?))", 'priority': 1, 'confidence': 0.99, 'locale': 'en_US', 'type': 'DATE', 'id': 'fdc63959-88e4-4859-bbed-7ba071d90593', 'example': 'Christmas, New Year, Thanksgiving Day, Memorial Day, etc.', 'groupAndRule': {'rule': 'holidaysRule', 'group': 'DateGroup'}}}
    result.append(NoTime)
    
    df_date= pd.DataFrame(columns = ['TemporalExpression', 'day', 'month', 'year'])
    df_date.head()
        
    df_date1=pd.concat([pd.DataFrame([elem['temporalExpression']], columns=['TemporalExpression']) for elem in result], ignore_index=True)
    df_date2=pd.concat([pd.DataFrame([(((elem['temporal'][0])['endDate'])['date'])['day']], columns = ['day']) for elem in result], ignore_index=True)
    df_date3=pd.concat([pd.DataFrame([(((elem['temporal'][0])['endDate'])['date'])['month']], columns = ['month']) for elem in result], ignore_index=True)
    df_date4=pd.concat([pd.DataFrame([(((elem['temporal'][0])['endDate'])['date'])['year']], columns = ['year']) for elem in result], ignore_index=True)
    df_date=df_date1.join(df_date2['day'])
    df_date=df_date.join(df_date3['month'])
    df_date=df_date.join(df_date4['year'])

    df_date["fullDate"] = df_date['day'].map(str) + '-' + df_date['month'].map(str) + '-' + df_date['year'].map(str)
    df_date.loc[df_date['TemporalExpression'] == 'No Time Identified', 'fullDate'] = 'No time identified'
    df_date=df_date.drop_duplicates(subset=['fullDate'], keep='first', inplace=False)

    return df_date['fullDate'].tolist()

def when_score(request):
    if request.method == 'POST':
        date = request.POST.get('date',None)

        datem = datetime.datetime.strptime(date, "%d-%m-%Y")
        giornoDataEstratta = datem.day
        meseDataEstratta = datem.month
        annoDataEstratta= datem.year
        
        today = datetime.date.today()
        giorno=(today.day)
        mese=(today.month)
        anno=(today.year)
        
        d1 = datetime.date(anno, mese, giorno)
        if (annoDataEstratta == 0 and meseDataEstratta==0 and giornoDataEstratta == 0):
            days_diff=("Unknown date")
            score=0
        else:
            d0 = datetime.date(annoDataEstratta, meseDataEstratta, giornoDataEstratta)
            delta = d1 - d0
            days_diff=(delta.days)
            if days_diff < 180:
                score=1
            elif (days_diff < 730 and days_diff > 179):
                score=1.15
            else:
                score=1.35
        request.session['when_score'] = score
    # return render(request,'home/who.html')
    return HttpResponseRedirect('what')


def where_analysis(request,text):
    locations = []
    nlp_wk = spacy.load('xx_ent_wiki_sm')
    doc = nlp_wk(text)
    locations.extend([[ent.text, ent.start, ent.end] for ent in doc.ents if ent.label_ in ['LOC']])
    locations.append(['No Location identified', 0, 0])
    
    df = pd.DataFrame(locations, columns=['Location', 'start','end'])
    df.head()
    
    df2 = pd.DataFrame([['No Location identified', 0, 0, 0, 0, 0, 0 ,0]], columns=['Location', 'start','end','address','coordinates','latitude', 'longitude', 'altitude'])

    if len(locations) > 0:
        locator = geopy.geocoders.Nominatim(user_agent="mygeocoder")
        geocode = RateLimiter(locator.geocode, min_delay_seconds=1)
        df["address"] = df["Location"].apply(geocode)
        df['coordinates'] = df['address'].apply(lambda loc: tuple(loc.point) if loc else None)
        df[['latitude', 'longitude', 'altitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
        df.latitude.isnull().sum()
        df = df[pd.notnull(df["latitude"])]
        df=df.drop_duplicates(subset=['Location'], keep='first', inplace=False)
        df=df.append(df2)
        return df['Location'].tolist(),df
    
def where_score(request):
    where(request)
    df = where.df

    if request.method == 'POST':
        answers = request.POST.get('nloc', None)
        address = request.POST.get('loc',None)
       
        latitudineLocalita=float(df.loc[df['Location'] == answers, 'latitude'])
        longitudineLocalita=float(df.loc[df['Location'] == answers, 'longitude'])
       
        if (latitudineLocalita == 0 and longitudineLocalita == 0):
            distance=("No Location identified")
            score=0
        else:  
    # return HttpResponseRedirect('where')
        
            geolocator = Nominatim(user_agent="mygeocoder")
            local = [address]
            if address != '':
                location = geolocator.geocode(address)
                try:
                    latitudineIndirizzo=location.latitude
                    longitudineIndirizzo=location.longitude
                    if (latitudineIndirizzo != 0 and longitudineIndirizzo != 0):
                    
                            lat1=latitudineIndirizzo
                            lon1=-longitudineIndirizzo
                            lat2=latitudineLocalita
                            lon2=-longitudineLocalita
                            R = 6371 
                            dLat = (lat2-lat1)*(Math.pi/180)  
                            dLon = (lon2-lon1)*(Math.pi/180) 
                            a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat1*(Math.pi/180)) * Math.cos(lat2*(Math.pi/180)) * Math.sin(dLon/2) * Math.sin(dLon/2)
                            c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)) 
                            d = R * c 
                            distance=d
                            if distance < 1650:
                                score=1
                            elif(distance > 1649 and distance < 6000 ):
                                score=1.15
                            else:
                                score=1.35
                    else:
                        score = 1
                        distance=("No Location identified")
                except AttributeError:
                    score = 1
                    distance=("No Location identified")
                
            else: 
                score = 1
                distance=("No Location identified")

    # else:
    #     score = 0
    #     distance=("No Location identified")
    request.session['where_score'] = score
    print(score,distance)
    # return(distance, score)
    return HttpResponseRedirect('why')

def count_upper(text):
	if text != '':
		c_up=sum(1 for c in text if c.isupper())
		sum_letter=sum(1 for c in text)
		rep=c_up/sum_letter
		if (rep < (10*sum_letter)/100):
			score=1
		else:
			score=0
	else:
		score = 0
		rep = 0
	return(rep, score)

def searchG(textN):  
    query = textN
    
    j=search(query, num_results=5) 
    res_link={}
    link_tot=[]
    title_tot=[]
    
    for elem in j: 
        reqs = requests.get(elem)
		# using the BeaitifulSoup module 
        soup = BeautifulSoup(reqs.text, 'html.parser')

        for title in soup.find_all('title'):
            titolo=(title.get_text())
        link_tot.append(elem)
        title_tot.append(titolo)
        res_link['Link'] = link_tot
        res_link['Title'] = title_tot
	#print(res_link)
    df2 = pd.DataFrame([['No Link identified', 'No Title Identified']], columns=['Link', 'Title'])

    df = pd.DataFrame.from_dict(res_link, orient='columns') 
    df = df.drop_duplicates(subset=['Link'], keep='first', inplace=False)
    df = df.append(df2)
    df["LinkTitle"] = df['Link'].map(str) + ' || ' + df['Title'].map(str) 
    return df['LinkTitle'].tolist()

def ImageAnalize():
    scores=0
    path_1 = "media/articleImage/"
    if len(os.listdir(path_1)) != 0:
        onlyfiles = [i for i in listdir(path_1) if isfile(join(path_1, i))] 
        print(onlyfiles)
        contScore=0
        scorePar=0
        img_item = []
        not_approved= []
        for elem in onlyfiles:
            filePath = path_1 + elem
            searchUrl = 'http://www.google.com/searchbyimage/upload'
            multipart = {'encoded_image': (filePath, open(filePath, 'rb')), 'image_content': ''}
            response = requests.post(searchUrl, files=multipart, allow_redirects=False)
            fetchUrl = response.headers['Location']
			
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')  # Last I checked this was necessary.
            driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=options)

			#Specify Search URL 
            search_url=fetchUrl 
            driver.get(search_url.format())

            df = pd.DataFrame(columns = ['Link'])
            df2 = pd.DataFrame([['No Link identified']], columns=['Link'])

            ricResult=driver.find_elements_by_xpath("//a[contains(@class,'fKDtNb')]")
            totalricResults=len(ricResult)

            if totalricResults != 0:    
                for ei in ricResult:
                    ser_correlated = {elem : ei.text}
                    img_item.append(ser_correlated)
            
            linResult=driver.find_elements_by_css_selector('div.g')
            if len(linResult) != 0:
                for els in linResult:
                    link = els.find_element_by_tag_name("a")
                    href = link.get_attribute("href")
                    df = df.append({'Link': href}, ignore_index=True)
                    df2 = df2.append(df)
                    df2 = df2.drop_duplicates(subset=['Link'], keep='first', inplace=False)
                    # df2 = df2.drop(df2.index[len(linResult)])
                    not_approved.append(df2['Link'].tolist())
				
						


	# 						questions = [inquirer.List('linktit', message="Which of these is closest to the image?", choices=df2['Link'].tolist(),),]
	# 						answers = inquirer.prompt(questions)
					   	

	# 						if answers["linktit"] == "No Link identified":
	# 							scorePar = scorePar + 0
	# 							contScore = contScore +1

	# 						else:
	# 							scorePar = scorePar + 1
	# 							contScore = contScore +1
								
	# 					else:
	# 						scorePar= scorePar + 0
	# 						contScore= contScore + 1
	# 		else:
	# 			scorePar= scorePar + 0
	# 			contScore= contScore + 1
	# else:
	# 	scores = 1

	# if scores==0:
	# 	score = scorePar/contScore
	# else:
	# 	score=scores
	
    # if(not_approved != None):
        not_approved = list(itertools.chain.from_iterable(not_approved))
        not_approved = [x for x in not_approved if not isinstance(x, int)]
        not_approved = list(dict.fromkeys(not_approved))
        return img_item, not_approved   
    else:
        img_item = None
        not_approved= None
        return img_item, not_approved 
