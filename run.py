import warnings
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings('ignore')


def main():
    df = pd.read_csv('/workspaces/Text-Classifier-for-UtaPass-and-KKBOX/data/UtaPass-KKBOX-Reviews.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['rating'])
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    # encodings = tokenizer(
    #     df['content'].tolist(), 
    #     max_length=64, 
    #     padding=True, 
    #     truncation=True, 
    #     return_tensors="pt"
    # )
    # print(tokenizer.decode(encodings["input_ids"][0]))

    train_tokens = [tokenizer.tokenize(review) for review in train_df['content'].tolist()]
    test_tokens = [tokenizer.tokenize(review) for review in test_df['content'].tolist()]
    train_y = train_df['rating'].tolist()
    test_y = test_df['rating'].tolist()
    
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, ngram_range=(1, 3), max_features=4000)    
    train_X = tfidf.fit_transform(train_tokens)
    test_X = tfidf.transform(test_tokens)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_X.toarray(), train_y)
    accuracy = metrics.accuracy_score(test_y, clf.predict(test_X.toarray()))
    print(accuracy)


if __name__ == '__main__':
    main()