# recommend_challenges
This is for developing the model engine that recommends challenges.

New validated changes needs to be integrated into the ml_function.

Use 'python -m pipreqs.pipreqs' to generate the requirements.txt if the file is not there.

Run main.py to get recommendations for an arbitary user_id

The KNN is implemented in the ml-function/challenge_recommender file which is the integrated version because the KNN is not availiable here.

Run rank_all function to generate the recommendations for all the users

Run recommend(user_id) to ouput a recommendation for a single user.

For math and the understanding of the model read the first section of this tutorial. https://www.toptal.com/algorithms/predicting-likes-inside-a-simple-recommendation-engine

