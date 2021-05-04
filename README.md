# Collaborative-filtering
This Project was done as a part of the online course Machine Learning:From linear models to deep learning by mit from EDX.
A data matrix from netflix containing 1200 users and their ratings on the movies they watched out of 1200 movies was given and our task was to predict the ratings of the movies which are not rated by the users.so, the shape of the matrix was 1200*1200. 
Collaborative filtering was done by mdelling the users as gaussian mixtures and parameters were optimized with K-means and EM algorithm.
The MSE was about 0.5 for EM-algorith and 0.8 for K-means.
The helper functions in common.py was already provided by the instructors.The rest of the code was implemented by myself from scratch.
