import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ast import literal_eval

class CB:
    def __init__(self):
        # keep simscores at the end of recommendation
        self.keepsimscores = list()

        # we read the data bases and merge them together for suitability
        print("reading data..", end=' ')
        df1 = pd.read_csv("Databases/tmdb_5000_credits.csv")
        self.df2 = pd.read_csv("Databases/tmdb_5000_movies.csv")

        df1.columns = ['id', 'tittle', 'cast', 'crew']
        self.df2 = self.df2.merge(df1, on='id')
        print("done")


        # set the string features into corresponding objects
        print("constructing corresponding objects..", end=' ')
        features = ['cast', 'crew', 'keywords', 'genres', 'production_companies']
        for feature in features:
            self.df2[feature] = self.df2[feature].apply(literal_eval)
        print("done")


        # define new suitable director, cast, genres and keyword features
        print("defining new suitable features..", end=' ')
        self.df2['director'] = self.df2['crew'].apply(self.GetDirector)

        features = ['cast', 'keywords', 'genres', 'production_companies']
        for feature in features:
            self.df2[feature] = self.df2[feature].apply(self.GetList)
        print("done")


        # apply CleanData to remove spaces between names
        print("cleaning data..", end=' ')
        features = ['cast', 'keywords', 'director', 'genres', 'production_companies']

        for feature in features:
            self.df2[feature] = self.df2[feature].apply(self.CleanData)
        print("done")


        # now we can create "metadata soup" to feed our vectorizer
        print("creating and feeding the soup to vectorizer..", end=' ')
        self.df2['soup'] = self.df2.apply(self.CreateSoup, axis=1)


        # One important difference is that we use the CountVectorizer() instead of TF-IDF.
        # This is because we do not want to down-weight the presence of an actor/director if
        # he or she has acted or directed in relatively more movies. It doesn't make much intuitive sense.
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(self.df2['soup'])


        # compute similarity
        self.cosine_sim = cosine_similarity(count_matrix, count_matrix)
        print("done")


        # reset index of our main dataframe and construct reverse mapping as before
        print("constructing reverse title-index mapping ..", end=' ')
        self.df2 = self.df2.reset_index()
        self.indices = pd.Series(self.df2.index, index=self.df2['title'])
        print("done")
        print("ready to recommend")

    # FUNCTIONS
    def GetRecommend(self, titles):
        # get the indices of the titles
        idx = []
        for title in titles:
            idx.append(self.indices[title])

        # set similarity scores
        sim_scores = []
        for index in idx:
            sim_scores.append(list(enumerate(self.cosine_sim[index])))

        # sort the movies based on similarity scores
        for i in range(len(sim_scores)):
            sim_scores[i] = sorted(sim_scores[i], key=lambda x: x[1], reverse=True)

        # get first 10 movies from each
        for i in range(len(sim_scores)):
            sim_scores[i] = sim_scores[i][1:11]

        # get movie indices
        movie_indices = []
        for sim_score in sim_scores:
            movie_indices.append([i[0] for i in sim_score])

        # recommend
        recidx = []
        recommendations = []
        while len(recommendations) < 10:
            index = random.choice(range(len(titles)))
            movie = movie_indices[index].index(random.choice(movie_indices[index]))
            item = self.df2['title'].iloc[movie_indices[index][movie]]
            if item not in recommendations:
                recommendations.append(item)

                # keep the ids of the movies related and recommended
                recidx.append(tuple([titles[index], item, sim_scores[index][movie][1]]))


        # keep movies/scores
        self.keepsimscores = recidx

        # recommend
        return recommendations


    # FUNCTIONS TO EXTRACT INFO FROM EACH FEATURE
    # get director's name
    def GetDirector(self, x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    # return the top 3 elements
    def GetList(self, x):
        if isinstance(x, list):
            names = [i['name'] for i in x]

            # check if more than 3 elements exist
            if len(names) > 3:
                names = names[:3]
            return names

        # return empty list in case of missing data
        return []

    # convert the words into lowercase and strip the spaces
    def CleanData(self, x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # check if director exists, if not, return empty str
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ""

    def CreateSoup(self, x):
        str = ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
        if len(x['production_companies']) > 0:
            str += ' '.join(x['production_companies'])
        return str


