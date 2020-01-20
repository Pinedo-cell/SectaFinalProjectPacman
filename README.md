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

Second attempt has 4 different algorithms:

The first one has reach almost 1.600.000 steps training and we found that it was bugged. 

The second one was a fixed algorithm with the same bases as the previous one, but with less batch size and less steps. 

The third one was an implementation of the algorithm using the ram of our Pc, instead of using the GPU to train our models.

The fourth one was an implementation of the vanilla dqn method, that was trained with 10m of steps, and we obtain the best scores. 


___________________________________________________________________________________________________________________________________
# Aditional features to our Reinforcement Learning work: 

Apart of the second option we also implemented two other reinforcement learning methods: Deep Q-Learning and Double Q-Learning.
The deep Q-Learning is implemented and explained on the folder SectaPacmanAditionalFeatures with our working code and explanation.
The double Q-Learning is not implemented yet because of some errors at the implementation of the code, as the code is not implemented for the Pacman.
Also we gathered all the colab implementations in the SectaPacmanColab folder. There is an environment to make Pacman OpenAI work.
