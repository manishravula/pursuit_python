import numpy as np
from src.global_defs import Point2D

"""
static const Point2D VARIABLE_IS_NOT_USED MOVES[NUM_MOVES] = {Point2D(1,0),Point2D(-1,0),Point2D(0,1),Point2D(0,-1),Point2D(0,0)};
RIGHT,
    LEFT,
    UP,
    DOWN,
    NOOP,
"""

ACTION_TO_MOVES = {1:Point2D(1,0),2:Point2D(-1,0),3:Point2D(0,1),4:Point2D(0,-1),5:Point2D(0,0)}
ACTIONS_TO_ACTIONPROBS = {1:[1,0,0,0,0],2:[0,1,0,0,0],3:[0,0,1,0,0],4:[0,0,0,1,0],5:[0,0,0,0,1]}
MOVES_TO_ACTIONS = {'10':1,'-10':2,'01':3,'0-1':4,'00':5}
