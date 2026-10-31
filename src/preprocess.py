import html

def clean_tweet(text):
    text = html.unescape(text)
    text = text.lower()
    words = text.split()
    cleaned = []
    for word in words:
        if word.startswith("http") or word.startswith("www."):
            cleaned.append("URL")
        elif word.startswith("@"):
            cleaned.append("USER")
        elif word.startswith("#"):
            cleaned.append(word[1:])
        else:
            cleaned.append(word)
    return " ".join(cleaned)

def preprocess(df):
    df = df.copy()
    df["clean_tweet"] = df["tweet"].apply(clean_tweet)
    return df
