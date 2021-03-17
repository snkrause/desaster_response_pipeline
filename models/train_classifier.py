import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    """
    

    Parameters
    ----------
    database_filepath : STRING
        path of the database for messages and their classification.

    Returns
    -------
    X : array
        messages column.
    Y : array
        36 categoriy labels.
    category_names : list
        names of the categories.

    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages',engine)
    
    #prepare feature and classes as arrays
    X = df['message'].values
    Y = df[df.columns[4:40]].values
    category_names=df[df.columns[4:40]].columns
    
    return X, Y, category_names


def tokenize(text):
    """
    

    Parameters
    ----------
    text : STRING
        text string to be tokenized.

    Returns
    -------
    clean_tokens : LIST
        cleaned up list of the individual tokens of the text string.

    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ",text)

    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    

    Returns
    -------
    pipeline : CLASS
        ready to use NLP pipeline.

    """
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(estimator=DecisionTreeClassifier(splitter='random', criterion='entropy')))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Testing the model with test data and printing out the average precision and f1-score
    for all categories.

    Parameters
    ----------
    model : CLASS
        full NLP pipeline.
    X_test : array
        messages column.
    Y_test : array
        36 categoriy labels.
    category_names : list
        names of the categories.

    Returns
    -------
    None.

    """
    #predict test values
    Y_pred=model.predict(X_test)
    f1_score=[]
    precision=[]
    classif=[]
    #check for each classification column the precision and f1-score
    for n in range(Y_pred.shape[1]):
        s=classification_report(Y_test[:,n], Y_pred[:,n],output_dict=True)
        classif.append(category_names[n])
        precision.append(s['1.0']['precision'])
        f1_score.append(s['1.0']['f1-score'])
    #print result    
    print('avg_precision:'+str(sum(precision)/len(precision))+', avg_f1-score:'+str(sum(f1_score)/len(f1_score)))
    
    #form a dataframe with the results and export it as excel file
    results=pd.DataFrame(classif,columns=['classifier'])
    results['precision']=precision
    results['f1-score']=f1_score
    results['model']="DecisionTreeClassifier"
    results.to_excel("results.xlsx", index=False)      

def save_model(model, model_filepath):
    """
    

    Parameters
    ----------
    model : CLASS
        full NLP pipeline.
    model_filepath : STRING
        path where to save the pkl file to store the fitted model.

    Returns
    -------
    None.

    """
    # Open the file to save as pkl file
    model_pkl = open(model_filepath, 'wb')
    pickle.dump(model, model_pkl)
    # Close the pickle instances
    model_pkl.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()