import re
import pandas as pd
import numpy as np
# cwd = os.getcwd()
# print(cwd)
# files = os.listdir(os.path.join(cwd, r'rec\challenge_category'))  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
df_challenges = pd.DataFrame(columns = ['challenge_id','challenge', 'category'])
# os.mkdir('rec\challenge_category')
cat=''
with open(r'ml_new\rec\daily_challenges_category.txt','r', encoding='utf-8') as f:
    for count, i in enumerate(f):
        if re.search(r'^After.+',i):
            df_challenges.loc[count] = [count,i.strip(),cat.strip()]
        else:
            cat = i
df_challenges.to_csv(r'ml_new\rec\challenges_data.csv',header=['challenge_id','challenge', 'category'], index=False)
df = pd.DataFrame(index=range(2000),columns=range(3))
df[0] = np.random.randint(40,size=(2000,))
df[1] = np.random.choice(df_challenges.challenge_id,(2000,))
df[2] = np.random.choice([1,0], (2000,))
df[3] = np.random.choice([1], (2000,))
df.to_csv(r'ml_new\rec\user_challenges_likes_v2.csv', header = ['user_id', 'challenge_id', 'likes','done'], index=False)