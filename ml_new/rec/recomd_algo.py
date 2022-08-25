import pandas as pd
import pickle
import random
import numpy as np
from collections import defaultdict
challenge_likes_path = r'ml_new\rec\user_challenges_likes_v2.csv' # path to user's liked paths
challenge_data_path = r'ml_new\rec\challenges_data.csv' # type of challenge, challenge ID, and info about challenge
cat = pd.read_csv(challenge_data_path) # challenge data
cat_list = cat['category'].unique() # categories of challenges
df_likes = pd.read_csv(challenge_likes_path) # user's likes of a challenge
df_likes['score'] = np.where(df_likes['likes']==1, 1,0) # determine the score of the challenge based on whether user likes or not
users = df_likes['user_id'].unique() # get all users
challenges = df_likes['challenge_id'].unique() # get all challenges


'''
Function used to filter our rows based on conditions. For example, can filter
user ID, challenge ID, and score
@param df - table to be filtered
@param filter_cols - column to be filtered
@param filter_conditions - condition to filter columns
@return - the filtered table
'''
def filtering(df, filter_cols,filter_conditions):
    for col, condition in zip(filter_cols, filter_conditions):
        df = df[df[col]==condition]
    return df

'''
Fills out data for each users. For each user, want their liked and disliked challenges. In
the dictionary, has a tuple with two sets: challenges user likes and dislikes
'''
user_dict = {} # user dictionary - used to store liked and disliked challenges
for user in users:
    liked_challenges = set(filtering(df_likes,['user_id','score'],[user,1])['challenge_id'])
    unliked_challenges = set(filtering(df_likes,['user_id','score'],[user,0])['challenge_id'])
    user_dict[user]=(liked_challenges,unliked_challenges)

'''
Fills out data for each challenge. For each challenge, want users who liked and disliked it. In
the dictionary, has a tuple with two sets: users who liked and who disliked the challenge
'''
challenge_dict={} # similar to user_dict, but reversed. Has two sets: users who liked and disliked the challenge
for challenge in challenges:
    user_liked = set(filtering(df_likes,['challenge_id','score'],[challenge,1])['user_id'])
    user_unliked = set(filtering(df_likes,['challenge_id','score'],[challenge,0])['user_id'])
    challenge_dict[challenge]=(user_liked,user_unliked)

# ORIGINAL CODE: user_most_inter = sorted([i for i in df_likes['user_id'].unique()], key=lambda x: filtering(df_likes, ['user_id'],[x]).shape[0], reverse=True)[:int(len(users)*0.15) if len(users) > 50 else int(len(users)*0.5)]
# user_most_inter - stores most active users (people who interact with challenges the most)
user_most_inter = sorted(list(df_likes['user_id'].unique()), key=lambda x: filtering(df_likes, ['user_id'], [x]).shape[0], reverse=True)
user_most_inter = user_most_inter[: int(len(users) * 0.15) if len(users) > 50 else int(len(users) * 0.5)]
print(user_most_inter)

'''
Function calculates similarities between two users. 
@param user1 - first user to compare
@param user2 - second user to compare
@return - similarity index between user1 and user2
'''
def similairty_index(user1, user2):
    # all the challenges that the two have rated
    all_sets = [user_dict[user1][0], user_dict[user1][1], user_dict[user2][0], user_dict[user2][1]]

    # challenges they both like
    like_like_intersection = len(user_dict[user1][0].intersection(user_dict[user2][0]))

    # challenges they both dislike
    unlike_unlike_intersection = len(user_dict[user1][1].intersection(user_dict[user2][1]))

    # challenges they disagree on
    like_unlike_intersection = len(user_dict[user1][1].intersection(user_dict[user2][0])) + len(user_dict[user1][0].intersection(user_dict[user2][1]))

    # length (number) of challenges they agree on
    union = len(set().union(*all_sets))

    # calculated by: challenges the both like + challenges they both dislike - challenges they disagree on / total # challenges
    return float(like_like_intersection + unlike_unlike_intersection - like_unlike_intersection) / union

'''
Function predicts if a user will like a certain challenge
@param user_id - user to check
@param challenge_id - challenge to check
@return - score from 1 to -1. >0 means user likes, <0 means user dislikes
'''
def user_like_predict(user_id, challenge_id):
    # all users who have liked challenge
    user_liked = set(filtering(df_likes, ['challenge_id', 'score'], [challenge_id, 1])['user_id'])

    # all users who have disliked challenge
    user_unliked = set(filtering(df_likes, ['challenge_id', 'score'], [challenge_id, 0])['user_id'])

    
    sum_similarity_user_liked = 0.0
    sum_similarity_user_unliked = 0.0
    # calculates similarity for current user to all other users who have liked the challenge
    for user in user_liked:
        sum_similarity_user_liked += similairty_index(user_id, user)
    # calculates similarity for current user to all other users who have disliked the challenge
    for user in user_unliked:
        sum_similarity_user_unliked += similairty_index(user_id, user)
    # total number of users who have rated the challenge
    num_user_rated = len(user_liked) + len(user_unliked)
    # if no one has rated challenge, means it is new and return 0.5 as a placeholder score 
    # Only recommend this challenge to most active users
    if num_user_rated == 0:
         return 0.5 if user_id in user_most_inter else -1
    # return the calculated prediction by: 
    # (similarity to users who liked challenge - similarity to users who disliked challenge) / total numbers of users who rated challenge
    return float(sum_similarity_user_liked - sum_similarity_user_unliked) / num_user_rated

'''
Function predicts user preference on all challenges
@param user_id - user to predict for
@return - sorted prediction score of the user's preference for all challenges
'''
def ranking(user_id):
    df = pd.read_csv(challenge_data_path)
    challenges = df['challenge_id'].unique()
    rankings = [(challenge, user_like_predict(user_id, challenge), cat[cat['challenge_id'] == challenge]['category'].values) for challenge in challenges]
    return sorted(rankings, key=lambda tup: tup[1], reverse=True)

'''
Function to save files in a pkl format
@param data - file to be saved
@param path - endpoint to save pkl file
'''
def save_pickle(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

'''
Function to load pkl file
@param path - endpoint to retrieve file
@return the pkl file
'''
def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)  

'''
Function ranks each user's preference on all challenges by calling the ranking 
function on all users. Saves this data as a pkl file.
'''
def rank_all():
    df = pd.read_csv(challenge_likes_path)
    users = df['user_id'].unique()
    df_ranking = {user: ranking(user) for user in users}
    # For new users, we want to recommend them the most popular challenges since
    # we don't have much data on them. Save it to dictionary with new_user as 
    # the key.
    most_liked = sorted(challenge_dict, key=lambda item: len(challenge_dict[item][0]), reverse=True)
    df_ranking['new_user'] = [(challenge, 0.8,cat[cat['challenge_id'] == challenge]['category'].values) for challenge in most_liked]
    save_pickle(df_ranking, r'ml_new\rec\saved_ranking_w_cat.p')

'''
Function recommends to user with a challenge from each category
@param user - user to recommend to
@return - list of one challenge from each category
'''
def recommend(user):
    df = pd.read_csv(challenge_likes_path)
    users = df['user_id'].unique()
    ranking = load_pickle(r'ml_new\rec\saved_ranking_w_cat.p')
    res = defaultdict(list)
    rank = ranking['new_user'] if user not in users else ranking[user]
    for score in rank:
        if len(res[score[2][0]]) > 0:
            if score[1] > 0.0:
                res[score[2][0]].append(score)
        else:
            res[score[2][0]].append(score)
    return [random.choice(value) for key, value in res.items()]


# def testing():
#     df_likes
#     return

# Run this function everyday to update user preferences on all challenges
rank_all()
print(*recommend(3), sep='\n')
# use knn to process the questionnaire of the user to determine which users are similar to 
# him/her and then recommend challgenges base on that. 