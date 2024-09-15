import streamlit as st
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
abbreviation_dict = {
    'wif': 'with',
    'hv': 'have',
    'EV': 'Electric Vehicle',
    'shld': 'should',
    'i.g.': 'for example',
    'btw': 'by the way',
    'bc': 'because'
}
# Preprocess Functions
def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    replaced_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return replaced_text

def replace_supersub(text):
    super_regex = re.compile(r'[\u00B2\u00B3\u00B9\u2070-\u2079]')
    sub_regex = re.compile(r'[\u2080-\u2089]')
    text = super_regex.sub(lambda m: str(unicodedata.numeric(m.group())), text)
    text = sub_regex.sub(lambda m: str(unicodedata.numeric(m.group())), text)
    return text

# Function to extract year, brand, and name from the car title
def extract_year_brand_name(title):
    if isinstance(title, str):
        year = re.search(r'^\d{4}', title).group(0) if re.search(r'^\d{4}', title) else None
        brand = re.search(r'^\d{4}\s+(\w+)', title).group(1) if re.search(r'^\d{4}\s+(\w+)', title) else None
        car_name = re.search(r'^\d{4}\s+\w+\s+(.*)', title).group(1).strip() if re.search(r'^\d{4}\s+\w+\s+(.*)', title) else None
        return year, brand, car_name
    else:
        return None, None, None

# Function to get sentiment using VADER
def get_sentiment(review):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(review)['compound']

# Load dataset
df = pd.read_csv("processed_reviews.csv", encoding='ISO-8859-1')
df_reviews = df.dropna()

# Extract car features (year, brand, name)
df_reviews[['Car_Year', 'Car_Brand', 'Car_Name']] = df_reviews['Vehicle_Title'].apply(lambda x: pd.Series(extract_year_brand_name(x)))

# Add a preprocessed reviews column (you can modify preprocessing as needed)
df_reviews['reviews_cleaned'] = df_reviews['Review'].apply(replace_abbreviations).apply(replace_supersub)
df_reviews['vader_ss'] = df_reviews['reviews_cleaned'].apply(get_sentiment)
df_reviews['vader_ss_normalize'] = df_reviews['vader_ss'].apply(lambda x: 1 if x >= 0 else 0)
# Function to rank the cars based on sentiment score
def rank_cars(df, top_n=5):
    ranked_df = df.sort_values(by='vader_ss_normalize', ascending=False).head(top_n)
    return ranked_df

# Streamlit app layout
st.title('Car Sentiment Analysis & Recommendation System')

# User input for review
st.header('Write a review or choose a car')
input_choice = st.radio("Choose how you'd like to get recommendations:", ('Write your own review', 'Select car features'))

# Option 1: Write your own review
if input_choice == 'Write your own review':
    user_review = st.text_area("Enter your car review here:", "")
    if st.button("Submit Review"):
        sentiment_score = get_sentiment(user_review)
        st.write(f"Sentiment Score: {sentiment_score}")

# Option 2: Select car features
if input_choice == 'Select car features':
    car_brand = st.selectbox("Select Car Brand", df_reviews['Car_Brand'].unique())
    car_name = st.selectbox("Select Car Name", df_reviews[df_reviews['Car_Brand'] == car_brand]['Car_Name'].unique())
    
    # Filter and rank cars based on selected features and sentiment
    filtered_cars = df_reviews[(df_reviews['Car_Brand'] == car_brand) & (df_reviews['Car_Name'] == car_name)]
    ranked_cars = rank_cars(filtered_cars)
    
    st.write("Top 5 Cars with Highest Sentiment Score:")
    st.dataframe(ranked_cars[['Car_Year', 'Car_Brand', 'Car_Name', 'vader_ss', 'Review']])

# Show the word cloud of top reviews (optional)
st.header('Word Cloud of Top Reviews')
wordcloud_button = st.button('Generate Word Cloud')

if wordcloud_button:
    from wordcloud import WordCloud
    wordcloud = WordCloud().generate(' '.join(df_reviews['reviews_cleaned'].dropna()))
    
    # Plot the wordcloud
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
