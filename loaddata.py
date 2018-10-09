
###获取电影序号数组
def countMovie():
    movieid_List=[]
    movie_file = open('D:/毕业论文/ml-1m/movies.dat', 'r', encoding='gb18030', errors='ignore')
    for line in movie_file:
        line = line.strip('\n').split('::')
        if line[0] not in movieid_List:
            movieid_List.append(line[0])
    return movieid_List

###为年龄性别电影种类标号
def voc_to_int():
    gender_count=0
    age_count=1
    movie_genre_count=1
    movie_title_count=1
    movie_title_int={}
    movie_genre_int={}
    gender_int={}
    age_int={}
    user_file=open('D:/毕业论文/ml-1m/users.dat','r',encoding='gb18030',errors='ignore')
    movie_file=open('D:/毕业论文/ml-1m/movies.dat','r',encoding='gb18030',errors='ignore')
    for line in user_file:
        line=line.strip('\n').split('::')
        if line[1] not in gender_int:
            gender_int[line[1]]=gender_count
            gender_count = gender_count + 1
        if line[2] not in age_int:
            age_int[line[2]]=age_count
            age_count = age_count + 1
    for line in movie_file:
        line = line.strip('\n').split('::')
        line[2]=line[2].split('|')
        for token in line[2]:
            if token not in movie_genre_int:
                movie_genre_int[token]=movie_genre_count
                movie_genre_count=movie_genre_count+1
        """
        pattern = re.compile('[^0-9()]+')
        result=pattern.findall(line[1])
        result[0]=result[0].split(' ')
        for item in result[0]:
            if item not in movie_title_int:
                movie_title_int[item]=movie_title_count
                movie_title_count=movie_title_count+1
        """
    return movie_genre_count,age_count,gender_count,movie_genre_int,age_int,gender_int


###获得用户的观看历史
def output_watch_History():
    watch_history={}
    rating_file = open('D:/毕业论文/ml-1m/ratings.dat', 'r', encoding='gb18030', errors='ignore')
    for line in rating_file:
        line=line.strip('\n').split('::')
        if line[0] not in watch_history:
            watch_history[line[0]]=[]
        if line[0] in watch_history:
            watch_history[line[0]].append(line[1])
    return watch_history

###计算关键词的出现概率
def count_probobality():
    genre_Pro={}
    movie_file = open('D:/毕业论文/ml-1m/movies.dat', 'r', encoding='gb18030', errors='ignore')
    for line in movie_file:
        line = line.strip('\n').split('::')
        line[2] = line[2].split('|')
        for token in line[2]:
            if token not in  genre_Pro:
                genre_Pro[token]=0
            if token in genre_Pro:
                genre_Pro[token]=genre_Pro[token]+1
    for token in genre_Pro.keys():
        genre_Pro[token]=genre_Pro[token]/3883
    return genre_Pro

def genre_to_Num():
    movie_genre_count, age_count, gender_count, movie_genre_int, age_int, gender_int = voc_to_int()
    genre_Pro=count_probobality()
    embed_Dic = {}
    for token in genre_Pro.keys():
        if token in movie_genre_int:
              embed_Dic[movie_genre_int[token]]=genre_Pro[token]
    return embed_Dic









