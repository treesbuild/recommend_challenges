from math import nan
import os
import re
import pandas as pd
import numpy as np
cwd = os.getcwd()
print(cwd)
files = os.listdir(os.path.join(cwd, r'rec\challenge_category'))  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
# df_challenges = pd.DataFrame(columns = ['challenge_id','challenge'])
# # os.mkdir('rec\challenge_category')
# with open(r'rec\daily_challenges_category.txt','r', encoding='utf-8') as f:
#     for count, i in enumerate(f):
#         if re.search(r'^After.+',i):
#             df_challenges.loc[count] = [count,i.strip()]
#         elif count != 0:
#             path = ''.join(['rec\challenge_category\\', re.sub('|'.join([" ", "/",r"\\"]), '_', i).strip(),'.csv'])
#             print(path)
#             df_challenges.to_csv(path, index=False)
            

df = pd.DataFrame(index=range(200),columns=range(3))
df[0] = np.random.randint(5,size=(200,))
df[1] = np.random.choice(df_challenges.challenge_id,(200,))
df[2] = np.random.choice([1,0], (200,))
df.to_csv(r'rec\user_challenges_likes_v2.csv', header = ['user_id', 'challenge_id', 'likes'], index=False)