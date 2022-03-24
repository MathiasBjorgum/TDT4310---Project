import pandas as pd

def clean_stock_data(stock_df: pd.DataFrame) -> pd.DataFrame:
    '''Adds labels for degree of change in the stock price, based on `Open` and `Close` columns.'''
    stock_df["change"] = (stock_df["Close"] - stock_df["Open"]) / stock_df["Open"]
    stock_df["change_cat"] = pd.cut(x = stock_df["change"], bins = [-0.2, -0.02, 0, 0.02, 0.2], labels = [0, 1, 2, 3])
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df = stock_df.drop(labels = ["Open", "High", "Low", "Close", "Volume", "Adj Close"], axis=1)
    return stock_df

def clean_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    '''Combines all news data headlines with the same date, and adds it all to one column.'''
    returning_df = news_df.groupby("Date")
    returning_df = returning_df["News"].apply(list)
    returning_df = returning_df.reset_index()
    returning_df["Date"] = pd.to_datetime(returning_df["Date"])
    return returning_df

def clean_and_combine(stock_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = clean_stock_data(stock_df)
    news_df = clean_news_data(news_df)

    return pd.merge(stock_df, news_df, how="outer", on="Date")