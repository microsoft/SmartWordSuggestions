# -*- coding:utf8 -*-
import urllib3
urllib3.disable_warnings()
import requests
import re
import random
import time
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import threading

res = {}

def get_one_page(url):

    for test in range(2000):
        requests.adapters.DEFAULT_RETRIES = 5
        session = requests.session()
        session.keep_alive = False
        session.headers = {"User-Agent":\
                    random.choice(["Mozilla/5.0 (Windows NT 10.0; WOW64)",
                  'Mozilla/5.0 (Windows NT 6.3; WOW64)',
                  'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
                  'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
                  'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.95 Safari/537.36',
                  'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; rv:11.0) like Gecko)',
                  'Mozilla/5.0 (Windows; U; Windows NT 5.2) Gecko/2008070208 Firefox/3.0.1',
                  'Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070309 Firefox/2.0.0.3',
                  'Mozilla/5.0 (Windows; U; Windows NT 5.1) Gecko/20070803 Firefox/1.5.0.12',
                  'Opera/9.27 (Windows NT 5.2; U; zh-cn)',
                  'Mozilla/5.0 (Macintosh; PPC Mac OS X; U; en) Opera 8.0',
                  'Opera/8.0 (Macintosh; PPC Mac OS X; U; en)',
                  'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.12) Gecko/20080219 Firefox/2.0.0.12 Navigator/9.0.0.6',
                  'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Win64; x64; Trident/4.0)',
                  'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; Trident/4.0)',
                  'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.2; .NET4.0C; .NET4.0E)',
                  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Maxthon/4.0.6.2000 Chrome/26.0.1410.43 Safari/537.1 ',
                  'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; InfoPath.2; .NET4.0C; .NET4.0E; QQBrowser/7.3.9825.400)',
                  'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0 ',
                  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.92 Safari/537.1 LBBROWSER',
                  'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0; BIDUBrowser 2.x)',
                  'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/3.0 Safari/536.11',
                  "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
                  "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
                  "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
                  "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
                  "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
                  "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
                  "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
                  "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
                  "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
                  "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
                  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
                  "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
                  "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
                  "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
                  "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
                  "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
                  "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
                  "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
                  "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
                  "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
                  "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
                  "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
                  "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
                  "UCWEB7.0.2.37/28/999",
                  "NOKIA5700/ UCWEB7.0.2.37/28/999",
                  "Openwave/ UCWEB7.0.2.37/28/999",
                  "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
                  # iPhone 6：
                  "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25"]), "Connection": "close"}
        session.verify = False
        re = session.get(url)
        if re.status_code == 200:
            response = re.text
            return response
        elif re.status_code == 404:
                return 0
        else: time.sleep(1)

def get_a_word(word):

    html = get_one_page("https://www.merriam-webster.com/thesaurus/%s" % re.sub(' ', '\%20', word))
    soup = BeautifulSoup(html, 'lxml')

    poses = soup.find_all("div", class_="row entry-header thesaurus") + soup.find_all("div", class_="row entry-header thesaurus long-headword")
    # poses = [re.sub("\s", '', re.sub("Save Word", "", re.sub(word, '', i.text))) for i in poses]
    poses = [i.find_all("span", class_='fl')[0].text for i in poses]

    pos_thesaurus = soup.find_all("div", class_="thesaurus-entry")
    res = []

    for i in range(len(pos_thesaurus)):

        senses = pos_thesaurus[i].find_all("span", class_="sb-0")
        for sense in senses:
            sense_res = {}
            sense_sents = re.split('\s{2,}', sense.find('span', class_='dt').text)
            sense_res['sense'] = sense_sents[1]
            sense_res['pos'] = poses[i]
            if len(sense_sents) > 2: sense_res['example_sentence'] = re.sub("\n", "", sense_sents[2])
            for span in sense.find_all('span', class_=re.compile("thes-list.*?")):
                title = re.sub(' ' + word, '', span.find('p').text)
                words = [i.text for i in span.find_all('a')]
                sense_res[title] = words
            res.append(sense_res)
    return res
    # return {word: res}

def get_page_num(char):

    html = get_one_page('https://www.merriam-webster.com/browse/thesaurus/%s' % char)
    return len(re.findall('<li>', re.findall('class=\"entries\"(.*?)class=\"pagination', html, re.S)[0], re.S))

def get_content_page(url):

    html = get_one_page(url)
    return [re.findall('<span>(.*?)</span>', i)[0] for i in re.findall('<li class=\"col-6 col-lg-4\">(.*?)</li>', re.findall('d-flex flex-wrap align-items-baseline row(.*?)</div>', html, re.S)[0], re.S)]

def run(word):
    res[word] = get_a_word(word)

if __name__ == '__main__':

    # res = get_a_word("appals")
    # print(json.dumps(res, indent=2))

    total_words = []

    for i in 'abcdefghijklmnopqrstuvwxyz':
        for j in range(get_page_num(i)):
            words = get_content_page('https://www.merriam-webster.com/browse/thesaurus/%s/%d' % (i, j+1))
            total_words += words
    print(len(total_words))

    for word in tqdm(total_words):
        while threading.active_count()>1000: time.sleep(1)
        threading.Thread(target=run, args=(word,)).start()

    while threading.active_count()>2: 
        print(threading.active_count())
        time.sleep(1)
    f = open('thesaurus.json', 'w')
    json.dump(res, f, indent=4)
