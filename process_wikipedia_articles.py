#%%
import mwparserfromhell
import lxml.etree as ET
import pandas as pd
import sqlite3

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

def insert_articles_batch(db_path, articles):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO articles (title, text) VALUES (?, ?)', articles)
    conn.commit()
    conn.close()

def parse_wikipedia_dump(file_path, db_path, filter_df = None):
    # print("Beginning to parse")
    articles = []
    batch_size = 1000  # Adjust batch size based on system capabilities
    processed_article_count = 0
    saved_article_count = 0
    titles_set = set(filter_df["Title"])
    for event, elem in ET.iterparse(file_path, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.10/}page'):
        title = elem.findtext('{http://www.mediawiki.org/xml/export-0.10/}title')
        
        if isinstance(filter_df, pd.DataFrame):
            if title.strip() not in titles_set:
                processed_article_count += 1
                if processed_article_count%100000 == 0:
                    # print(title)
                    print(f'Processed {processed_article_count} articles!')
                elem.clear()
                continue

        revision = elem.find('{http://www.mediawiki.org/xml/export-0.10/}revision')
        text = revision.findtext('{http://www.mediawiki.org/xml/export-0.10/}text')
        
        clean_text = process_article(title, text)
        articles.append((title, clean_text))

        # print(title)
        if len(articles) >= batch_size:
            # break
            insert_articles_batch(db_path, articles)
            saved_article_count += len(articles)
            print(f'Saved {saved_article_count} articles!')
            articles = []
        
        elem.clear()
    
    # Insert any remaining articles
    if articles:
        insert_articles_batch(db_path, articles)

def process_article(title, text):
    '''Process and clean article text, returning plain text.'''
    wikicode = mwparserfromhell.parse(text)
    clean_text = wikicode.strip_code().strip()
    return clean_text

def read_txt(filename):
    '''Necessary as pandas read csv does nor work properly with the titles file.'''

    with open(filename, 'r') as file:
        lines = file.readlines()
    df = pd.DataFrame(lines, columns=['Text'])
    df['Text'] = df['Text'].str.strip()#.str.lower()

    return df

if __name__ == '_main_':
    views_df = read_txt('/home/david/Downloads/enwiki-2023-pv.txt')
    page_titles_df = read_txt('/home/david/Downloads/enwiki-2023.titles.txt')

    page_views_df = pd.DataFrame({'Title': page_titles_df['Text'], 'Views': views_df['Text']})
    page_views_df['Views'] = page_views_df['Views'].astype(float).astype(int)
    page_views_df['Title'] = page_views_df['Title'].astype("string")
    more_than_100000_views_df = page_views_df[page_views_df['Views']>100000].copy()
    more_than_100000_views_df.to_csv("more_than_100000_views_df.csv")

    db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
    create_db(db_path)
    dump_file_path = '/media/david/WDBLUE8TB/data/enwiki-20240501-pages-articles.xml'
    parse_wikipedia_dump(dump_file_path, db_path, more_than_100000_views_df)
#%%