from rec.recomd_algo import recommend
from rec.recomd_algo import ranking
cha = recommend(1)
ran = ranking(1)
print(cha)
print(ran)
print([item for item in ran if item[0] == cha])
