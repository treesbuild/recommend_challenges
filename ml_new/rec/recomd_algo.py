from unicodedata import category
import pandas as pd
import pickle
import numpy as np

cat = pd.read_csv(r'ml_new\rec\challenges_data.csv')
cat_list = cat['category'].unique()
df_likes = pd.read_csv(r'ml_new\rec\user_challenges_likes_v2.csv')
df_likes['score'] = np.where(np.logical_and(df_likes['likes']==1, df_likes['done']==1), 1,0)
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
    df = pd.read_csv(r'ml_new\rec\user_challenges_likes_v2.csv')
    challenges = df['challenge_id'].unique()
    rankings = [(challenge, user_like_predict(user_id, challenge), cat[cat['challenge_id'] == challenge]['category'].values) for challenge in challenges]
    return sorted(rankings, key=lambda tup: tup[1], reverse=True)


def save_pickle(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)  


def rank_all():
    df = pd.read_csv(r'ml_new\rec\user_challenges_likes_v2.csv')
    users = df['user_id'].unique()
    df_ranking = {user: ranking(user) for user in users}
    most_liked = sorted(challenge_dict, key=lambda item: len(challenge_dict[item][0]), reverse=True)
    df_ranking['new_user'] = [(challenge, 0.8,cat[cat['challenge_id'] == challenge]['category'].values) for challenge in most_liked]
    save_pickle(df_ranking, r'ml_new\rec\saved_ranking_w_cat.p')


def recommend(user):
    df = pd.read_csv(r'ml_new\rec\user_challenges_likes_v2.csv')
    users = df['user_id'].unique()
    ranking = load_pickle(r'ml_new\rec\saved_ranking_w_cat.p')
    ans = []
    cat_added = set()
    rank = ranking['new_user'] if user not in users else ranking[user]
    for score in rank:
        if score[2][0] not in cat_added:
            ans.append(score)
            cat_added.add(score[2][0])
    return ans


# def testing():
#     df_likes
#     return

# rank_all()
print(*recommend(30), sep='\n')
