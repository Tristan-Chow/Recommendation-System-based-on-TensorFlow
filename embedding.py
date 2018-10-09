from loaddata import *
from numpy import *
import pandas as pd

from merge_sheet import *
genre_Pro=count_probobality()
embed_Dic=genre_to_Num()
mergetable=open('D:/毕业论文/finaldata.csv')
merge_Table=pd.read_csv(mergetable)

def addlist(list1,list2):
    return [list1[i]+list2[i] for i in range(min(len(list1),len(list2)))]

def movie_genre_embed():
    movie_genre_Embed = {}
    for lineNo in movie_table.index:
        line = movie_table.loc[lineNo]
        genre_Embed = [0] * 18
        for item in line['genre']:
            if item in embed_Dic:
                genre_Embed[item - 1] = embed_Dic[item]
        movie_genre_Embed[line['movieID']] = genre_Embed
    return movie_genre_Embed


def genre_Embedding():
    movie_genre_Embed=movie_genre_embed()
    for lineNo in merge_Table.index:
        line=merge_Table.loc[lineNo]
        merge_Table.at[lineNo,'genre']=movie_genre_Embed[line['movieID']]
        result=[0]*18
        for movie in eval(line['watch_history']):
            movie = int(movie)
            result = addlist(result,movie_genre_Embed[movie])
        merge_Table.at[lineNo,'watch_history']=result
    return merge_Table













