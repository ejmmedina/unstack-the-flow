## Appendix A. Data Wrangling


```python
#preliminaries

import os
from bs4 import BeautifulSoup
import requests
import time
import re
import numpy as np
import pandas as pd
import sqlite3
```

### Appendix A.1. Stackexchange Archive

The dataset used for this analysis is collected from the stackexchange repository: https://archive.org/details/stackexchange. Specifically, the file stackoverflow.com-Posts.7z which contains an XML file of all posts in stackoverflow was downloaded. Due to its large size (13.6G), the file was split into multiple files using `split -l` from `git bash`.

The XML file contains millions of rows where one row would contain the following information:

```xml
  <row Id="4" PostTypeId="1" AcceptedAnswerId="7" CreationDate="2008-07-31T21:42:52.667" Score="617" ViewCount="40863" Body="&lt;p&gt;I want to use a track-bar to change a form's opacity.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;This is my code:&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;decimal trans = trackBar1.Value / 5000;&#xA;this.Opacity = trans;&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;When I build the application, it gives the following error:&lt;/p&gt;&#xA;&#xA;&lt;blockquote&gt;&#xA;  &lt;p&gt;Cannot implicitly convert type &lt;code&gt;'decimal'&lt;/code&gt; to &lt;code&gt;'double'&lt;/code&gt;.&lt;/p&gt;&#xA;&lt;/blockquote&gt;&#xA;&#xA;&lt;p&gt;I tried using &lt;code&gt;trans&lt;/code&gt; and &lt;code&gt;double&lt;/code&gt; but then the control doesn't work. This code worked fine in a past VB.NET project.&lt;/p&gt;&#xA;" OwnerUserId="8" LastEditorUserId="6786713" LastEditorDisplayName="Rich B" LastEditDate="2018-07-02T17:55:27.247" LastActivityDate="2019-01-17T13:39:48.937" Title="Convert Decimal to Double?" Tags="&lt;c#&gt;&lt;floating-point&gt;&lt;type-conversion&gt;&lt;double&gt;&lt;decimal&gt;" AnswerCount="13" CommentCount="1" FavoriteCount="46" CommunityOwnedDate="2012-10-31T16:42:47.213" />
```

Each subfile (from the large file) would then be processed into an SQL database using `projdbload.py` with the code shown below:

```python
import sqlite3
import psycopg2
import pandas as pd
import re
import os
from xml.etree import ElementTree
import click
import random


@click.group()
def main():
    pass


@main.command('pathdbload')
@click.argument('pathname', type=str)
@click.argument('tablename', type=str)
@click.argument('database', type=str)
@click.option('--filt', type=str, default='row', help='filter')
def FiletoDBLoad(pathname, tablename, database, filt='row'):
    """Load the XML files under `pathname` and store the attributes in the
    table `tablename` under `database`.
    
    Parameters
    ----------
    pathname : str
        filepath leading to directory containing the XML files to be processed
    tablename : str
        name of the SQL table to store the data in
    database : str
        filename of the SQL database
    filt : str, optional
        filters the XML file based on the filt query. default value is row
    """
    files = []
    
    # Get the files under the directory "pathname"
    for r, d, f in os.walk(pathname):
        for file in f:
            os.system('echo "Will process this file "'+file)
            files.append(os.path.join(r, file))

    # Initialize the database
    conn = sqlite3.connect(database)
    curs = conn.cursor()

    for filename in files:
        # Store the current file into a temp file
        os.system('echo "Processing File "'+filename)
        os.system(
            'echo "<?xml version=\\\"1.0\\\" encoding=\\\"utf-8\\\"?>" > .temp.txt')
        head = '"<'+tablename.lower()+'>"'
        os.system('echo '+head+' >> .temp.txt')
        # Filter the XML file according to the query filt
        os.system('cat '+filename+' |grep "'+filt.replace('"', r'\"') +
                  '" >> .temp.txt')
        tail = '"</'+tablename.lower()+'>"'
        os.system('echo '+tail+' >> .temp.txt')

        # Open the temp file
        with open('.temp.txt', 'r', encoding='utf-8') as f:

            lines = f.read().splitlines()
            flag = True
            # Get the minimum number of columns to use as header from 5 sampled rows/elements
            for i in random.sample(range(2, len(lines)), 5):
                header_temp = lines[i]
                # Since display name and user ID are both unique identifier of users:
                header_temp = header_temp.replace('DisplayName', 'UserId')
                header_temp = re.findall(r'(\S*?)=', header_temp)
                if flag:
                    header = header_temp
                    flag = False
                    continue
                if len(header_temp) < len(header):
                    header = header_temp
        # Parse the temp file and store the attribute values corresponding to the headers obtained
        tree = ElementTree.parse('.temp.txt')
        xml_data = tree.getroot()
        table = []
        for elem in xml_data:
            row = {}
            for j in header:
                try:
                    col = str(f'{header.index(j):02d}')+'_'+j
                    row[col] = elem.attrib[j]
                except:
                    continue
            table.append(row)
        df_table = pd.DataFrame(table)
        df_table.columns = [i[3:] for i in df_table.columns]
        df_table.to_sql(tablename, con=conn, if_exists='append', index=False)
        # Move the finished files to a different folder
        os.system('mv ' + filename + ' ' + pathname +
                  '/../backup/' + pathname.split('/')[-1])


if __name__ == '__main__':
    main()
```

Note that the minimum number of columns were selected, i.e. the attributes that aren't generally found among the posts were ignored, e.g. `ParentId` which is contingent on being an answer post (since the parent must be a question post) or `ClosedDate` which appears only if the question is closed, will not be stored as a column entry in the database. The attributes chosen to be stored into the database are only those that are common or are relevant to the analysis.

A sample command line code used to process the files in the folder `Posts` and store the attributes of the xml files within that to the file `stackoverflow.db` is:


```python
!nohup python projdbload.py pathdbload Posts PostsQuestions stackoverflow.db --filt PostTypeId=\"1\" &
```

The --filt option was used to filter the posts such that the PostTypeId is equal to 1, corresponding to the questions (contrary to the PostTypeId=2 corresponding to answers).

After storing all the files into the database `stackoverflow.db`, SQL queries can be performed to obtain the necessary data.


```python
conn = sqlite3.connect(
    '/mnt/processed/private/msds2020/lt13/database/stackoverflow.db')
cur = conn.cursor()

cur.execute(f"""SELECT DISTINCT * FROM PostsQuestions 
            WHERE tags LIKE "%python%"
            LIMIT 10""")

names = list(map(lambda x: x[0], cur.description))
pd.DataFrame(cur.fetchall(), columns=names).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>PostTypeId</th>
      <th>AcceptedAnswerId</th>
      <th>CreationDate</th>
      <th>Score</th>
      <th>ViewCount</th>
      <th>Body</th>
      <th>OwnerUserId</th>
      <th>LastEditorUserId</th>
      <th>LastEditDate</th>
      <th>LastActivityDate</th>
      <th>Title</th>
      <th>Tags</th>
      <th>AnswerCount</th>
      <th>CommentCount</th>
      <th>FavoriteCount</th>
      <th>CommunityOwnedDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>337</td>
      <td>1</td>
      <td>342</td>
      <td>2008-08-02T03:35:55.697</td>
      <td>67</td>
      <td>7625</td>
      <td>&lt;p&gt;I am about to build a piece of a project th...</td>
      <td>111</td>
      <td>2336654</td>
      <td>2016-12-30T12:56:21.493</td>
      <td>2019-05-22T00:27:38.800</td>
      <td>XML Processing in Python</td>
      <td>&lt;python&gt;&lt;xml&gt;</td>
      <td>12</td>
      <td>1</td>
      <td>7</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>469</td>
      <td>1</td>
      <td>3040</td>
      <td>2008-08-02T15:11:16.430</td>
      <td>38</td>
      <td>2655</td>
      <td>&lt;p&gt;I am using the Photoshop's javascript API t...</td>
      <td>147</td>
      <td>1997093</td>
      <td>2016-12-22T03:53:45.467</td>
      <td>2016-12-22T03:53:45.467</td>
      <td>How can I find the full path to a font from it...</td>
      <td>&lt;python&gt;&lt;macos&gt;&lt;fonts&gt;&lt;photoshop&gt;</td>
      <td>4</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>502</td>
      <td>1</td>
      <td>7090</td>
      <td>2008-08-02T17:01:58.500</td>
      <td>40</td>
      <td>13577</td>
      <td>&lt;p&gt;I have a cross-platform (Python) applicatio...</td>
      <td>147</td>
      <td>63550</td>
      <td>2011-04-08T11:42:03.807</td>
      <td>2016-03-25T13:53:13.470</td>
      <td>Get a preview JPEG of a PDF on Windows?</td>
      <td>&lt;python&gt;&lt;windows&gt;&lt;image&gt;&lt;pdf&gt;</td>
      <td>3</td>
      <td>0</td>
      <td>13</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>535</td>
      <td>1</td>
      <td>541</td>
      <td>2008-08-02T18:43:54.787</td>
      <td>54</td>
      <td>8671</td>
      <td>&lt;p&gt;I am starting to work on a hobby project wi...</td>
      <td>154</td>
      <td>7232508</td>
      <td>2018-05-14T17:46:14.650</td>
      <td>2018-05-14T17:46:14.650</td>
      <td>Continuous Integration System for a Python Cod...</td>
      <td>&lt;python&gt;&lt;continuous-integration&gt;&lt;extreme-progr...</td>
      <td>7</td>
      <td>0</td>
      <td>13</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>594</td>
      <td>1</td>
      <td>595</td>
      <td>2008-08-03T01:15:08.507</td>
      <td>39</td>
      <td>42723</td>
      <td>&lt;p&gt;There are several ways to iterate over a re...</td>
      <td>116</td>
      <td>116</td>
      <td>2016-10-14T18:15:27.420</td>
      <td>2016-10-15T20:47:11.027</td>
      <td>cx_Oracle: How do I iterate over a result set?</td>
      <td>&lt;python&gt;&lt;sql&gt;&lt;database&gt;&lt;oracle&gt;&lt;cx-oracle&gt;</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



### Appendix A.2. Web Scraping
While the data from the stackexchange archive is complete, the project requires web scraping. And so, the top stackoverflow questions were scraped, particularly the information stored in the XML file.

Some preliminaries before scraping are the proxy settings and the request header to avoid being blocked and minimize the damage if you get blocked.


```python
# Proxy servers for web scraping

os.environ['HTTP_PROXY'] = 'http://3.112.188.39:8080'
os.environ['HTTPS_PROXY'] = 'https://3.112.188.39:8080'

def get_links(url, headers=None):
    """Get the links of the stackoverflow questions under `url`
    """
    resp = requests.get(url, headers=headers)

    resp_soup = BeautifulSoup(resp.text, 'lxml')

    links = [i.get('href')
             for i in resp_soup.select('div.summary a.question-hyperlink')]
    return links

# Request headers

req_header = '''accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3
accept-encoding: gzip, deflate, br
accept-language: en-US,en;q=0.9
cache-control: max-age=0
cookie: _ga=GA1.2.1223690127.1554361835; __qca=P0-1087024619-1554361834541; __gads=ID=95575f8f21b13b6a:T=1554361834:S=ALNI_MYzrAP4MO9xO3-RuodMxFKsUvXijA; notice-ctt=4%3B1554364020909; prov=5a1477ef-3607-2d87-c83b-4dadc0666ec0; _gid=GA1.2.229546230.1560849365; acct=t=sHvGyo1FYK9EJrJ2RTE92i%2bijUmMLbgd&s=ODXb%2b%2f3uMcMKfe3z1cbrgAlLUsX%2bAdVM; _gat=1
referer: https://stackoverflow.com/
upgrade-insecure-requests: 1
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'''

headers = {i.lower(): j for i, j in re.findall(
    r'(.*?): (.*?)$', req_header, re.M)}  # re.M --> multiline
url = 'https://stackoverflow.com/?tab=month'

links = get_links(url, headers)
```


```python
def scrape_stackoverflow(links, t=5):
    """Scrape each link in the list of links for the attributes stored in the
    XML file from stackexchange archive. An optional parameter t sets the 
    sleep time after each iteration.
    """
    data = []
    for link in links:
        url_2 = 'https://stackoverflow.com/' + link
        retry_count = 0
        while retry_count <= 20:  # Limit the number of retries
            try:
                resp_2 = requests.get(url_2, headers=headers)
                if resp_2.status_code == 200:
                    break
                else:
                    print("Red status code")
                    time.sleep(5)
                    retry_count += 1
            except Exception as e:
                print("Pause for 5 seconds before requesting again")
                time.sleep(5)
                retry_count += 1
        else:
            print("Too many retries")
            break

        # Scrape the data
        resp_soup2 = BeautifulSoup(resp_2.text, 'lxml')

        title = resp_soup2.select_one('div#question-header > h1 > a').text
        tags = [i.text for i in resp_soup2.select(
            'div.post-taglist > div > a')]
        tags = ''.join([f'<{i}>' for i in tags])

        # post id
        row_ID = int(re.findall(r'/questions/(\d+)/', link)[0])
        post_type_ID = 1
        accepted_ans_ID = resp_soup2.findAll(
            'div', {'itemprop': 'acceptedAnswer'})
        if accepted_ans_ID:
            accepted_ans_ID = accepted_ans_ID[0]['data-answerid']
        else:
            accepted_ans_ID = ''

        # post statistics
        question_stats = resp_soup2.select('div.module.question-stats b')
        creation_date = question_stats[0].time['datetime']
        view_count = int(re.findall(
            r'[0-9,]+', question_stats[1].text)[0].replace(',', ''))
        last_act_date = ''
        if len(question_stats) == 3:
            last_act_date = question_stats[2].a['title'].replace(
                ' ', 'T').replace('Z', '')

        #score and body
        score = int(resp_soup2.select_one(
            'div.js-vote-count.grid--cell.fc-black-500.fs-title.grid.fd-column.ai-center').text)
        body = ''.join(list(map(str, resp_soup2.select('div.post-text p'))))

        # user id and editor id
        user_dets_main = resp_soup2.select_one(
            'div.mt16.pt4.grid.gs8.gsy.fw-wrap.jc-end.ai-start')
        user_dets = user_dets_main.select('div.user-details')
        owner_ID = re.findall(r'/users/(\d+)/', user_dets[-1].a['href'])[0]
        if len(user_dets) == 2:
            edit_date = resp_soup2.select(
                'div.question div.user-action-time')[0].span['title'].replace(' ', 'T').replace('Z', '')
            if user_dets[0].has_attr('itemprop'):
                editor_ID = owner_ID
            else:
                try:
                    editor_ID = re.findall(
                        r'/users/(\d+)/', user_dets[0].a['href'])[0]
                except TypeError:
                    editor_ID = ''
        else:
            edit_date = ''
            editor_ID = ''

        # answer and comment count
        ans_count = int(resp_soup2.select_one(
            'div.subheader.answers-subheader').h2['data-answercount'])
        comm = resp_soup2.select('div.question div.comments ul')
        if comm[0]['class'][0] == 'close-as-off-topic-status-list':
            comm.pop(0)
        comm_hidden = int(comm[0][
                          'data-remaining-comments-count'])
        comm_shown = len(resp_soup2.select('div.question span.comment-copy'))
        comm_count = comm_hidden + comm_shown

        data.append((row_ID, post_type_ID, accepted_ans_ID, creation_date,
                     score, view_count, body, owner_ID, editor_ID, edit_date,
                     last_act_date, title, tags, ans_count, comm_count))
        time.sleep(t)
        clear_output()

        count += 1
        links.remove(link)
    return data
```


```python
data = scrape_stackoverflow(links, t=5)
df = pd.DataFrame(data, columns=['row Id', 'PostTypeId', 'AcceptedAnswerId',
                                 'CreationDate', 'Score', 'ViewCount', 'Body',
                                 'OwnerUserId', 'LastEditorUserId', 'LastEditDate',
                                 'LastActivityDate', 'Title', 'Tags', 'AnswerCount',
                                 'CommentCount']).astype(str)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>row Id</th>
      <th>PostTypeId</th>
      <th>AcceptedAnswerId</th>
      <th>CreationDate</th>
      <th>Score</th>
      <th>ViewCount</th>
      <th>Body</th>
      <th>OwnerUserId</th>
      <th>LastEditorUserId</th>
      <th>LastEditDate</th>
      <th>LastActivityDate</th>
      <th>Title</th>
      <th>Tags</th>
      <th>AnswerCount</th>
      <th>CommentCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>56738380</td>
      <td>1</td>
      <td>56738667</td>
      <td>2019-06-24T14:08:14</td>
      <td>110</td>
      <td>12753</td>
      <td>&lt;p&gt;Since C++ 17 one can write an &lt;code&gt;if&lt;/cod...</td>
      <td>4850111</td>
      <td>63550</td>
      <td>2019-06-26T17:36:50</td>
      <td>2019-07-09T13:34:20</td>
      <td>Most elegant way to write a one-shot 'if'</td>
      <td>&lt;c++&gt;&lt;if-statement&gt;&lt;c++17&gt;</td>
      <td>8</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>56807112</td>
      <td>1</td>
      <td>56812838</td>
      <td>2019-06-28T12:38:46</td>
      <td>82</td>
      <td>7707</td>
      <td>&lt;p&gt;Today I started to receive this error with ...</td>
      <td>5790492</td>
      <td>1032372</td>
      <td>2019-06-28T18:42:18</td>
      <td>2019-07-08T14:14:43</td>
      <td>Xcode ERROR ITMS-90783: “Missing bundle displa...</td>
      <td>&lt;xcode&gt;&lt;testflight&gt;&lt;fastlane&gt;&lt;appstoreconnect&gt;</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56866458</td>
      <td>1</td>
      <td>56866529</td>
      <td>2019-07-03T08:58:55</td>
      <td>59</td>
      <td>4166</td>
      <td>&lt;p&gt;I read that in C++17 we can initialize vari...</td>
      <td>10858827</td>
      <td>3628942</td>
      <td>2019-07-10T10:23:55</td>
      <td>2019-07-10T10:27:18</td>
      <td>Initializing variables in an “if” statement</td>
      <td>&lt;c++&gt;&lt;c++17&gt;</td>
      <td>6</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56692117</td>
      <td>1</td>
      <td>56692435</td>
      <td>2019-06-20T18:43:16</td>
      <td>124</td>
      <td>6596</td>
      <td>&lt;p&gt;When using the same code, simply changing t...</td>
      <td>9419412</td>
      <td>63550</td>
      <td>2019-06-27T12:21:35</td>
      <td>2019-06-27T12:21:35</td>
      <td>Why is C++ initial allocation so much larger t...</td>
      <td>&lt;c++&gt;&lt;c&gt;&lt;benchmarking&gt;</td>
      <td>2</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56642369</td>
      <td>1</td>
      <td>56642520</td>
      <td>2019-06-18T05:40:29</td>
      <td>34</td>
      <td>4973</td>
      <td>&lt;p&gt;I updated 'android.support:appcompat-v7' to...</td>
      <td>11212074</td>
      <td>11212074</td>
      <td>2019-07-10T14:09:29</td>
      <td>2019-07-10T14:09:29</td>
      <td>Android Material and appcompat Manifest merger...</td>
      <td>&lt;android&gt;&lt;react-native&gt;&lt;react-native-android&gt;&lt;...</td>
      <td>13</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


