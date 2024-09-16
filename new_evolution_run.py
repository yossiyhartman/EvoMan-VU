################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
from evoman.environment import Environment

# ENVIRONMENT INIT
"""
def __init__(self,
    experiment_name='test',
    multiplemode="no",           # yes or no
    enemies=[1],                 # array with 1 to 8 items, values from 1 to 8
    loadplayer="yes",            # yes or no
    loadenemy="yes",             # yes or no
    level=2,                     # integer
    playermode="ai",             # ai or human
    enemymode="static",          # ai or static
    speed="fastest",             # normal or fastest
    inputscoded="no",            # yes or no
    randomini="no",              # yes or no
    sound="off",                  # on or off
    contacthurt="player",        # player or enemy
    logs="on",                   # on or off
    savelogs="yes",              # yes or no
    clockprec="low",
    timeexpire=3000,             # integer
    overturetime=100,            # integer
    solutions=None,              # any
    fullscreen=False,            # True or False
    player_controller=None,      # controller object
    enemy_controller=None,      # controller object
    use_joystick=False,
    visuals=False):
"""


# initializes environment with ai player and static enemies
experiment_name ='new_test_solution'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

 # load controller from text file
PLAYER_CONTROLLER =np.loadtxt('solutions_new/test_one.txt')
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')

for en in range(1):
    env = Environment(experiment_name=experiment_name,
                      enemymode='static',
                      speed="normal",
                      sound="off",
                      fullscreen=True,
                      use_joystick=True,
                      playermode='ai',
                      logs="on",
                      visuals=True)
    env.update_parameter('enemies', [en])
    env.play(PLAYER_CONTROLLER)




## TEST NEW SOLUTION
"""
sol = np.loadtxt('solutions_new/new1.txt')
print('\n Loaded new solution and running game \n')
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')

# tests saved demo solutions for each enemy
for en in range(1, 9):
	
	#Update the enemy
	env.update_parameter('enemies',[en])

	env.play(sol)
"""