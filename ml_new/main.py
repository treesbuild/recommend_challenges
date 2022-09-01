from rec.recomd_algo import recommend
from rec.recomd_algo import rank_all
import time
start_time = time.time()
rank_all()
print(f"main--- {time.time() - start_time} seconds ---")
