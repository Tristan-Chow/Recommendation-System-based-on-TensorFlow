from loaddata import voc_to_int
from loaddata import output_watch_History
import pandas as pd

movie_genre_count, age_count, gender_count, movie_genre_int, age_int, gender_int=voc_to_int()
watch_history=output_watch_History()
def age_normalization(age_int):
    for key in age_int:
        age_int[key]=(int(key)-1)/(56-1)
    return age_int


def sheet_to_input():
    title = ['userID', 'gender', 'age', 'occupyation', 'zipcode']
    movie_title=['movieID','movie_name','genre']
    user_table = pd.read_csv('D:/毕业论文/ml-1m/users.dat', sep='::', names=title)
    movie_table=pd.read_csv('D:/毕业论文/ml-1m/movies.dat',sep='::',names=movie_title)
    new_age_int = age_normalization(age_int)
    for indexs in user_table.index:
        line = user_table.loc[indexs]
        if line['gender'] in gender_int:
            user_table.loc[indexs,'gender']=gender_int[line['gender']]
        if str(int(line['age'])) in new_age_int:
            user_table.loc[indexs,'age']=new_age_int[str(int(line['age']))]
    user_table['watch_history']=None
    for indexs in user_table.index:
        line=user_table.loc[indexs]
        if str(line['userID']) in watch_history:
            user_table.at[indexs,'watch_history']=watch_history[str(line['userID'])]
    for indexs in movie_table.index:
        line=movie_table.loc[indexs]
        newline=line['genre'].strip('\n').split('|')
        genreList = []
        for item in newline:
            if item in movie_genre_int:
                genreList.append(movie_genre_int[item])
        movie_table.at[indexs,'genre']=genreList
    return movie_table,user_table




