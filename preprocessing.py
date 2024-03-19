from typing import Iterable
import re

from unidecode import unidecode
import pandas as pd
import spacy


# funciones de preprocesamiento de dataframe
def tf_to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = pd.to_numeric(df[col])
    return df


def preprocess_df(df: pd.DataFrame, to_numeric_col: list[str]) -> pd.DataFrame:
    """convert columns to numeric, add length of text and convert created_date to datetime"""
    df = tf_to_numeric(df, to_numeric_col)
    df["len_text"] = df["comment_text"].str.len()
    df["created_date"] = pd.to_datetime(df["created_date"].str.slice(0, 10))
    return df


def sample_from_comments(df: pd.DataFrame, n: int = 5_000, seed: int = 42):
    """sample n rows from the train set"""
    return df.query("split=='train'").sample(n, random_state=seed)


def sample_with_quota(df: pd.DataFrame, n_by_quota: int = 5_000):
    quota_sampling = []
    n_samples = 5
    for k in range(n_samples):
        start = 1 / n_samples * k
        end = 1 / n_samples * (k + 1)
        quota_sampling.append(
            df.query(
                f" (toxicity > {start}) & (toxicity <= {end}) & (split=='train')"
            ).sample(n_by_quota)
        )

    return pd.concat(quota_sampling)


# funciones de procesamiento de texto
def to_spacy(corpus: Iterable, nlp: spacy.language.Language):
    """Convert text to spacy doc"""
    return list(nlp.pipe(corpus, n_process=1))


def filter_stopwords_and_len(doc: spacy.tokens.Doc, len_min):
    """Remove stopwords and tokens with length less than len_min"""
    return filter(lambda token: not token.is_stop and len(token) >= len_min, doc)


def lemmatize(doc: spacy.tokens.Doc):
    """Lemmatize text"""
    return filter(lambda token: token.lemma_, doc)
    # return [token.lemma_ for token in doc]


def get_text_from_doc(doc: spacy.tokens.Doc):
    """Get text from spacy doc"""
    return " ".join(token.text for token in doc)


def normalize(text: str):
    """Convert to lowercase and remove accents"""
    return unidecode(text).lower()


def remove_nonalpha(text: str) -> str:
    """Remove non-alphabetic characters"""
    return re.sub(r"[^a-z ]", " ", text)


def remove_doublespaces(text: str):
    """Remove extra whitespaces"""
    return " ".join(text.split())


def preprocess_text(doc, lemma: bool = False) -> Iterable:
    """Preprocess text"""
    doc_filter = filter_stopwords_and_len(doc, 3)
    doc_filter = lemmatize(doc_filter) if lemma else doc_filter
    text = get_text_from_doc(doc_filter)
    text = remove_nonalpha(text)
    text = remove_doublespaces(text)
    return text


def drop_missing_comments(df: pd.DataFrame):
    return df[df["comment_text"].notna()]


def drop_short_comments(df: pd.DataFrame, min_len: int = 4):
    return df[df["comment_text"].str.len() > min_len]
