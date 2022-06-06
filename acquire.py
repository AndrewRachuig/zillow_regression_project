import env
import pandas as pd
import os
import numpy as np

def get_zillow():
    '''
    This function acquires the requisite zillow data from the Codeup SQL database and caches it locally it for future use in a csv 
    document; once the data is accessed the function then returns it as a dataframe.
    '''

    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        query = '''
        SELECT 
            bedroomcnt, 
            bathroomcnt, 
            fullbathcnt
            calculatedbathnbr,
            calculatedfinishedsquarefeet,
            lotsizesquarefeet,
            regionidcity,
            regionidcounty,
            regionidzip,
            roomcnt,
            yearbuilt, 
            transactiondate, 
            fips,
            taxvaluedollarcnt
        FROM 
            properties_2017 
        JOIN
            propertylandusetype USING (propertylandusetypeid)
        JOIN
            predictions_2017 USING (parcelid)
        Where
            propertylandusedesc = 'Single Family Residential' AND 
            transactiondate LIKE '2017-%%'      
        '''
        url = env.get_db_url('zillow')
        df = pd.read_sql(query, url)
        df.to_csv(filename, index = False)

        return df 

def minimum_sqr_ft(df):
    '''
    Function that takes in a dataframe and finds the minimum sq footage necessary given an input number of bathrooms and bedrooms.
    
    Returns a total minimum amount
    '''
    # min square footage for type of room
    bathroom_min = 10
    bedroom_min = 70
    
    # total MIN sqr feet
    total = (df.bathroomcnt * bathroom_min) + (df.bedroomcnt * bedroom_min)
    # return MIN sqr feet
    return total

def clean_sqr_feet(df):
    '''
    Takes in a dataframe finds the theoretical minimum sq footage given bathroom and bedroom inputs and compares that to the actual
    given sq footage.  
    Returns a dataframe where containing results only having an actual sq footage larger than the calculate minimum.
    '''
    # get MIN sqr ft
    min_sqr_ft = minimum_sqr_ft(df)
    # return df with sqr_ft >= min_sqr_ft
    # change 'sqr_ft' to whichever name you have for sqr_ft in df
    return df[df.calculatedfinishedsquarefeet >= min_sqr_ft]

def map_counties(df):
    # identified counties for fips codes 
    counties = {6037: 'los_angeles',
                6059: 'orange',
                6111: 'ventura'}
    # map counties to fips codes
    df.fips = df.fips.map(counties)
    df.rename(columns=({ 'fips': 'county'}), inplace=True)
    return df

def clean_zillow(df):
    '''
    This function takes in an uncleaned zillow dataframe and peforms various cleaning functions. It returns a cleaned zillow dataframe.
    '''
    # Gettring rid of unwanted columns
    df.drop(columns= ['regionidcity', 'regionidcounty', 'transactiondate', 'roomcnt'], inplace=True)

    # Getting rid of all nulls
    df = df.dropna()

    # Getting rid of invalid, wrong, or incorrectly entered data
    df = df[df.bedroomcnt != 0]

    # Getting rid of crazy outliers in the data.
    # Keeps the vast majority of data and makes it more applicable.
    df = df[df.lotsizesquarefeet < 300000]
    df = df[df.taxvaluedollarcnt < 4000000]
    df = df[df.calculatedfinishedsquarefeet < 8500]

    # Getting rid of nonsense entries where the house has a sq footage value smaller than a theoretical minimum
    df = clean_sqr_feet(df)

    # Changing the fips column in the dataframe to show actual counties represented by fips number
    df = map_counties(df)

    # Changing datatypes for selected columns to improve efficiency
    df.bedroomcnt = df.bedroomcnt.astype('uint8')
    df.bathroomcnt = df.bathroomcnt.astype('float16')
    df.calculatedbathnbr = df.calculatedbathnbr.astype('float16')
    df.yearbuilt = df.yearbuilt.astype('uint16')
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype('uint16')

    return df
    