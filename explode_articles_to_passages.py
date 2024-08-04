#%%
import gc
import pandas as pd
import sqlite3
from create_embeddings import split_into_passages

def insert_articles_batch(db_path, passages):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO article_passages (article_id, passage) VALUES (?, ?)', passages)
    conn.commit()
    conn.close()
#%%
conn = sqlite3.connect('/media/david/WDBLUE8TB/data/wikipedia_articles.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS article_passages (
        passage_id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER NOT NULL,
        passage TEXT
    );
''')
conn.commit()
conn.close()

db_path = '/media/david/WDBLUE8TB/data/wikipedia_articles.db'
conn = sqlite3.connect(db_path)
article_df = pd.read_sql_query("SELECT * FROM articles", conn)
conn.close()
#%%
article_df.drop('title', axis=1, inplace=True)
article_df['passages'] = article_df['text'].apply(split_into_passages)
passage_df = article_df.explode('passages')
del article_df  
gc.collect()
#%%
passage_df.rename(columns = {'passages': 'passage_text', 'id':'article_id'}, inplace = True)
passage_df.drop(['text'], axis=1, inplace=True)
gc.collect()
#%%
passage_tuples = [tuple(x) for x in passage_df.to_numpy()]

del passage_df  
gc.collect()
#%%
insert_articles_batch(db_path, passage_tuples)
# %%
