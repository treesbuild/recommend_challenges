import pandas as pd
import pickle
import numpy as np


df_likes = pd.read_csv(r'rec\user_challenges_likes_v2.csv')
users = df_likes['user_id'].unique()
challenges = df_likes['challenge_id'].unique()
user_dict = {}


def filtering(df, filter_cols,filter_conditions):
    for col, condition in zip(filter_cols, filter_conditions):
        df = df[df[col]==condition]
    return df


for user in users:
    liked_challenges = set(filtering(df_likes,['user_id','score'],[user,1])['challenge_id'])
    unliked_challenges = set(filtering(df_likes,['user_id','score'],[user,0])['challenge_id'])
    user_dict[user]=(liked_challenges,unliked_challenges)

challenge_dict={}


for challenge in challenges:
    user_liked = set(filtering(df_likes,['challenge_id','score'],[challenge,1])['user_id'])
    user_unliked = set(filtering(df_likes,['challenge_id','score'],[challenge,0])['user_id'])
    challenge_dict[challenge]=(user_liked,user_unliked)
#get the top 10 most liked challenges
most_liked = sorted(challenge_dict, key=lambda item: len(challenge_dict[item][0]), reverse=True)[:10]


def similairty_index(user1,user2):
    all_sets = [user_dict[user1][0],user_dict[user1][1] ,user_dict[user2][0], user_dict[user2][1]]
    like_like_intersection = len(user_dict[user1][0].intersection(user_dict[user2][0]))
    unlike_unlike_intersection = len(user_dict[user1][1].intersection(user_dict[user2][1]))
    like_unlike_intersection = len(user_dict[user1][1].intersection(user_dict[user2][0])) + \
        len(user_dict[user1][0].intersection(user_dict[user2][1]))
    union = len(set().union(*all_sets))
    return float(like_like_intersection+unlike_unlike_intersection-like_unlike_intersection) / union


def user_like_predict(user_id, challenge_id):
    user_liked = set(filtering(df_likes,['challenge_id','score'],[challenge_id,1])['user_id'])
    user_unliked = set(filtering(df_likes,['challenge_id','score'],[challenge_id,0])['user_id'])
    sum_similarity_user_liked = 0.0
    sum_similarity_user_unliked = 0.0
    for user in user_liked:
        sum_similarity_user_liked += similairty_index(user_id,user)
    for user in user_unliked:
        sum_similarity_user_unliked += similairty_index(user_id,user)
    num_user_rated = len(user_liked)+len(user_unliked)
    if num_user_rated == 0:
        return 0.1
    return float(sum_similarity_user_liked-sum_similarity_user_unliked) / num_user_rated


def ranking(user_id):
    df = pd.read_csv(r'rec\user_challenges_likes_v2.csv')
    challenges = df['challenge_id'].unique()
    users = df['user_id'].unique()
    if user_id not in users:
        return sorted(challenge_dict, key=lambda item: len(challenge_dict[item][0]), reverse=True)[:10]
    return sorted([(challenge, user_like_predict(user_id, challenge)) for challenge in challenges], key=lambda challenge: challenge[0], reverse=False)


def load_json(path):
    with open(path, 'r') as fp:
        return pickle.load(fp)

def save_pickle(data):
    with open(r'rec\saved_ranking.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)  

def rank_all():
    df = pd.read_csv(r'rec\user_challenges_likes_v2.csv')
    users = df['user_id'].unique()
    df_ranking = dict()
    for user in users:
        df_ranking[user] = ranking(user)
    # save_json(df_ranking)
    save_pickle(df_ranking)

def recommend(user):
    ranking = load_pickle(r'rec\saved_ranking.p')
    return np.random.choice(ranking[user][:10], 1)


print(ranking(3))

