from makesheet import *
import pandas as pd
movie_table,user_table=sheet_to_input()

def merge_table():
    title=['userID','movieID','rating','timestamp']
    rating_file=pd.read_csv('D:/毕业论文/ml-1m/ratings.dat',sep='::',names=title)
    result=pd.merge(rating_file,user_table,on="userID",how="left")
    final=pd.merge(result,movie_table,on="movieID",how="left")
    del final['zipcode']
    del final['rating']
    del final['movie_name']
    final['label'] = 1
    return final





