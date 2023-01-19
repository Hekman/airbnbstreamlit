import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
lemmatizer = nltk.stem.WordNetLemmatizer()

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

df = pd.read_csv('listings.csv')

df['description'] = df['description'].fillna('Unknown')

def clean_text(x):
  stop_words = stopwords.words('english')

  clean = re.compile('<.*?>')
  x = re.sub(clean, ' ', x)

  # remove punctuation
  x = x.translate(str.maketrans(' ', ' ', string.punctuation))

  # lowercase
  x = x.lower()
 
  # strip excessive whitespaces
  x = x.strip()
  
  # tokenize
  x = nltk.word_tokenize(x)

  # remove stopwords
  x = [token for token in x if not token in stop_words]

  # lemmatization and pass string back
  x = ' '.join([w for w in x])

  return x

df['clean_description'] = df['description'].apply(clean_text)

df['host_response_rate'] = df['host_response_rate'].str.replace('%', '')
df['host_response_rate'] = df['host_response_rate'].fillna(0)
df['host_response_rate'] = df['host_response_rate'].astype('int')

df.loc[df['host_response_rate'].between(0, 25), 'speed'] = '0%-25%'
df.loc[df['host_response_rate'].between(26, 50), 'speed'] = '26%-50%'
df.loc[df['host_response_rate'].between(51, 75), 'speed'] = '51%-75%'
df.loc[df['host_response_rate'].between(76, 100), 'speed'] = '76%-100%'

df['price'] = df['price'].fillna(0)
df['price'] = df['price'].str.replace('$', '', regex=True)
df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].astype('float')

df['host_since_date'] = pd.to_datetime(df['host_since'])
df['year'] = pd.DatetimeIndex(df['host_since_date']).year

st.title('Airbnb Dashboard')
st.header('A basic dashboard that uses an Airbnb dataset')

col1, col2, col3 = st.columns(3)

with col1:
  values = st.slider(
    'Select years',
    2009, 2022, (2009, 2022))

with col2:
  values2 = st.slider(
    'Select price range',
    0, 1000, (0, 1000))

with col3:
  options = st.multiselect(
      'What type of room?',
      ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'],
      ['Entire home/apt', 'Private room', 'Hotel room', 'Shared room'])

df_selection = df[(df['year'] >= values[0]) & (df['year'] <= values[1])]
df_selection = df_selection[(df_selection['price'] >= values2[0]) & (df_selection['price'] <= values2[1])]
df_selection = df_selection[df_selection['room_type'].isin(options)]


# calculate how it grows over time
df_response = df_selection.groupby('grade')['host_response_rate'].count()
df_timeline = df_selection.groupby('host_since_date')['host_response_rate'].count().cumsum()

col1, col2 = st.columns(2)

with col1:
  st.header('Response rate')
  st.bar_chart(df_response)

with col2:
  st.header('Hosts over time')
  st.line_chart(df_timeline)

col3, col4 = st.columns(2)

with col3:
  st.header('Locations of Airbnbs')
  st.map(df_selection)
with col4:
  st.header('Most used words in description')
  text = df_selection['clean_description'].str.cat(sep=' ')
  w = WordCloud().generate(text)
  img = plt.imshow(w)
  plt.axis("off")
  st.pyplot()
