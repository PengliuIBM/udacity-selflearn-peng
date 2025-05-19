import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline

# Function to preprocess data
def preprocess_data() -> pd.DataFrame:
    """
    Reads movie data from .csv files, maps column names, adds the "Original Language" column,
    and finally concatenates all data into one dataframe.
    """
    df_eng = pd.read_csv("data/movie_reviews_eng.csv")
    df_fr = pd.read_csv("data/movie_reviews_fr.csv")
    df_sp = pd.read_csv("data/movie_reviews_sp.csv")
    
    # Standardizing column names
    df_eng.columns = ["Title", "Year", "Synopsis", "Review"]
    df_fr.columns = ["Title", "Year", "Synopsis", "Review"]
    df_sp.columns = ["Title", "Year", "Synopsis", "Review"]
    
    # Add language identifiers
    df_eng['Original Language'] = 'en'
    df_fr['Original Language'] = 'fr'
    df_sp['Original Language'] = 'sp'
    
    # Combine into a single dataframe
    df = pd.concat([df_eng, df_fr, df_sp], ignore_index=True)
    return df

df = preprocess_data()

# Load translation models and tokenizers
fr_en_model_name = "Helsinki-NLP/opus-mt-fr-en"
es_en_model_name = "Helsinki-NLP/opus-mt-es-en"
fr_en_model = MarianMTModel.from_pretrained(fr_en_model_name)
es_en_model = MarianMTModel.from_pretrained(es_en_model_name)
fr_en_tokenizer = MarianTokenizer.from_pretrained(fr_en_model_name)
es_en_tokenizer = MarianTokenizer.from_pretrained(es_en_model_name)

# Function to translate text
def translate(text: str, model, tokenizer) -> str:
    """
    Function to translate a text using a model and tokenizer
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# TODO 4: Filter and translate French reviews
fr_reviews = df.loc[df['Original Language'] == 'fr', 'Review']  # Get French reviews
fr_reviews_en = fr_reviews.apply(lambda x: translate(x, fr_en_model, fr_en_tokenizer))  # Translate to English

# Update dataframe with translated French reviews
df.loc[df['Original Language'] == 'fr', 'Review'] = fr_reviews_en

# TODO 4: Filter and translate French synopsis
fr_synopsis = df.loc[df['Original Language'] == 'fr', 'Synopsis']  # Get French synopsis
fr_synopsis_en = fr_synopsis.apply(lambda x: translate(x, fr_en_model, fr_en_tokenizer))  # Translate to English

# Update dataframe with translated French synopsis
df.loc[df['Original Language'] == 'fr', 'Synopsis'] = fr_synopsis_en

# TODO 4: Filter and translate Spanish reviews
es_reviews = df.loc[df['Original Language'] == 'sp', 'Review']  # Get Spanish reviews
es_reviews_en = es_reviews.apply(lambda x: translate(x, es_en_model, es_en_tokenizer))  # Translate to English

# Update dataframe with translated Spanish reviews
df.loc[df['Original Language'] == 'sp', 'Review'] = es_reviews_en

# TODO 4: Filter and translate Spanish synopsis
es_synopsis = df.loc[df['Original Language'] == 'sp', 'Synopsis']  # Get Spanish synopsis
es_synopsis_en = es_synopsis.apply(lambda x: translate(x, es_en_model, es_en_tokenizer))  # Translate to English

# Update dataframe with translated Spanish synopsis
df.loc[df['Original Language'] == 'sp', 'Synopsis'] = es_synopsis_en


##rewrite below code to code framework as TODO5, TODO6,TODO7 at below.
# TODO 5: Update the code below
# load sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_classifier = pipeline("sentiment-analysis", model=model_name)

# TODO 6: Complete the function below
def analyze_sentiment(text, classifier):
    """
    function to perform sentiment analysis on a text using a model
    """
    result = classifier(text)
    return result[0]['label']  # Return sentiment label (e.g., 'POSITIVE' or 'NEGATIVE')

# TODO 7: Add code below for sentiment analysis
# perform sentiment analysis on reviews and store results in new column
df['Sentiment'] = df['Review'].apply(lambda x: analyze_sentiment(x, sentiment_classifier))

# export the results to a .csv file
df.to_csv("result/reviews_with_sentiment.csv", index=False)

print("Processing complete. Output saved as 'reviews_with_sentiment.csv'.")
