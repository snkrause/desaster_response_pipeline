# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    

    Parameters
    ----------
    messages_filepath : STRING
        Path of the file containing the messages.
    categories_filepath : STRING
        Path of the file containing the categories.

    Returns
    -------
    df : DATAFRAME
        Datatable with merged data.

    """
    
    #load both files into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #join both 
    df = pd.merge(messages,categories,on='id')

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-',expand=True)[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        
        # convert column from string to numeric
        categories[column] =  categories[column].astype('int64')
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)
    
    return df

    
def clean_data(df):
    """
    

    Parameters
    ----------
    df : DATAFRAME
        Newly loaded dataframe with categorized messages.

    Returns
    -------
    df : DATAFRAME
        Cleaned dataframe w/o duplicates, empty categories or empty rows.

    """
    #drop duplicates
    df=df.drop_duplicates(subset='message', keep='first')

    #drop rows with more than 90% nan values
    df=df.drop(df[df.isnull().mean(axis=1)>0.9].index)

    #fill nan values with 0
    df=df.fillna(0)
    
    #drop column "related" since it only has the value 1
    df=df.drop('related',axis=1)
    
    #drop rows with only '0' which are not classified
    df=df.loc[df[df.columns[4:40]].sum(axis=1)!=0].reset_index(drop=True)
    
    #drop all rows in which a category is neither 0 nor 1
    num_cols=df.select_dtypes(exclude='object').columns.drop('id')
    for col in num_cols:
        df.drop(df.loc[~df[col].isin([0,1])].index, inplace=True)
        
    #drop empty categories
    for col in num_cols:
        if sum(df[col])==0:
            df.drop([col], axis=1, inplace=True)
        else:
            continue
    
    #drop rows with more than 15 assigned classifications
    df.drop(df[df[df.columns[4:39]].sum(axis=1)>15].index, axis=0, inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    

    Parameters
    ----------
    df : DATAFRAME
        Cleaned dataframe with categorized messages.
    database_filename : STRING
        Path for the database in which the messages are stored.

    Returns
    -------
    None.

    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()