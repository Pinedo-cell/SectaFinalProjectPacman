# SectaFinalGroupProyect
## Inteligent Systems Final Group Practice

This practice aim is to create different algorithms in order to learn how teach and ai to play pacman.

# First attempt to Reinforcement learning: 
The website where we have found the archives and we have been following the tutorial of Reinforcement Learning is the following: https://www.cse.huji.ac.il/~ai/reinforcement/reinforcement.html

We have modified the following .py archives: 
valueIterationAgents.py	(A value iteration agent for solving known MDPs.)
qlearningAgents.py	(Q-learning agents for Gridworld, Crawler and Pac-Man)

To make Pacman Learn in the Pacman original grid game you should run the following .py with the commands below: 
python3 pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l originalClassic

___________________________________________________________________________________________________________________________________

# Second attempt to Reinforcement learning: 
We tried to train the algorithm but we found some mistakes and errors during our training. 

Second attempt has 4 different algorithms, the first one has reach almost 1.600.000 steps training and we found that it was bugged. 

The second one was a fixed algorithm with the same bases as the previous one, but with less batch size and less steps. 

The third one was an implementation of the algorithm using the ram of our Pc, instead of using the GPU to train our models.

The fourth one was an implementation of the vanilla dqn method, that was trained with 10m of steps, and we obtain the best scores. 


___________________________________________________________________________________________________________________________________
# Aditional features to our Reinforcement Learning work: 
