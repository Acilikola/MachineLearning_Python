'''
Advantages and Disadvantages of Content-Based Filtering
Advantages
Learns user's preferences

Highly personalized for the user

Disadvantages
Doesn't take into account what others think of the item, so low quality item
recommendations might happen

Extracting data is not always intuitive

Determining what characteristics of the item the user dislikes or
likes is not always obvious

...
let's take a look at how to implement Content-Based or
Item-Item recommendation systems. This technique attempts to figure out
what a user's favourite aspects of an item is, and then recommends items
that present those aspects.

In our case, we're going to try to figure out the input's favorite genres
from the movies and ratings given.
("ml-latest\movies.csv", "ml-latest\ratings.csv")
'''
# import required libraries
import pandas as pd
pd.set_option('display.expand_frame_repr', False) #show all columns
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# read csv files to their dataframes
# skip 'timestamp' column in ratings since we won't need it + save memory
movies_df = pd.read_csv('ml-latest/movies.csv')
ratings_df = pd.read_csv('ml-latest/ratings.csv', usecols=["userId","movieId","rating"])
print(movies_df.head())

print()

## let's remove year info from 'title' and instead put it on a seperate column
#Using regular expressions to find a year stored between parentheses
#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
print(movies_df.head())
##

print()

## let's also split the list of 'Genres' into an array for future use
#Every genre is separated by a | so we simply have to call the split function on |
movies_df['genres'] = movies_df.genres.str.split('|')
print(movies_df.head())
##

print()

## let's also use 'One Hot Encoding' technique to convert list of genres to
## a vector where each column corresponds to one possible value of the feature
'''
This encoding is needed for feeding categorical data.

In this case, we store every different genre in columns that contain
either 1 or 0. 1 shows that a movie has that genre and
0 shows that it doesn't.

Let's also store this dataframe in another variable since genres
won't be important for our first recommendation system.
'''
#Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
print(moviesWithGenres_df.head())
##

print()

print(ratings_df.head())

print()

# let's create an input user with some ratings, to recommend movies to
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)
print(inputMovies)

print()

'''
With the input complete, let's extract the input movies's ID's from
the movies dataframe and add them into it.

We can achieve this by first filtering out the rows that contain
the input movies' title and then merging this subset with the input
dataframe. We also drop unnecessary columns for the input to
save memory space.
'''
##Add movieId to input user
#Filtering out the movies by title
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
#If a movie you added in above isn't here, then it might not be in the original 
#dataframe or it might spelled differently, please check capitalisation.
print(inputMovies)

print()

# start learning input user's preferences in terms of genres
#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
print(userMovies)

print()

'''
We'll only need the actual genre table, so let's clean this up a bit
by resetting the index and dropping the movieId, title, genres and year columns.
'''
#Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop=True)
#Dropping unnecessary issues due to save memory and to avoid issues
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(userGenreTable)

print()

# turn each genre to weights by multiplying genre with input's review ratings
# and summing by column (operation is called a 'dot product' between a matrix
# and vector)
#Dot produt to get weights
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
#The user profile
print(userProfile)

print()

'''
now, we have weights for input user's every preference (genre). this is known
as 'User Profile'. we should start recommending now
'''
## extract genre table from original dataframe
#Now let's get the genres of every movie in our original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#And drop the unnecessary information
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
print(genreTable.head())
print(genreTable.shape)
##

print()

'''
With the input's profile and the complete list of movies and their genres
in hand, we're going to take the weighted average of every movie
based on the input profile and recommend the top 20 movies that most
satisfy it.
'''
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
#Sort our recommendations in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
#Just a peek at the values
print(recommendationTable_df.head())

print()

## FINAL RECOMMENDATION TABLE
print("FINAL RECOMMENDATION TABLE")
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
