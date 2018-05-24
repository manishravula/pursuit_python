import numpy as np
import logging
from collections import namedtuple
import src.global_defs as defs
import src.global_const as const
import experiments.configuration as config


"""
static const Point2D VARIABLE_IS_NOT_USED MOVES[NUM_MOVES] = {Point2D(1,0),Point2D(-1,0),Point2D(0,1),Point2D(0,-1),Point2D(0,0)};
RIGHT,
    LEFT,
    UP,
    DOWN,
    NOOP,
"""

dim = config.DIMENSIONS


def movePosition(position,movement):
    """
    Function to calculate how motion is performed in the 2D space
    :param position: defs.Point2d tuple
    :param movement: defs.Point2d tuple
    :return: defs.Point2d tuple of the final position
    """

    newPosition = defs.Point2D(0,0)
    newPosition.x = (position.x + movement.x)%dim
    newPosition.y = (position.y + movement.y)%dim

    if(newPosition.x < 0):
        newPosition.x += dim
    if(newPosition.y < 0):
        newPosition.y += dim

    return newPosition

def getDistanceToPoint(pos1,pos2):
    """
    Function to calculate distance between 2 points in the toroidal space
    :param pos1: defs.Point2d tuple
    :param pos2: defs.Point2d tuple
    :return: distance
    """
    delta = getDifferenceToPoint(pos1,pos2)
    return abs(delta.x)+abs(delta.y)



def getDifferenceToPoint(pos_begin,pos_end):
    """
    Difference in the 2D coordinates in the toroidal space
    :param pos1: defs.Point2d tuple
    :param pos2: defs.Point2d tuple
    :return: defs.Point2d tuple
    """
    delta = defs.Point2D(0,0)

    maxDx = 0.5*(dim)
    maxDy = 0.5*(dim)

    delta.x = pos_end.x  - pos_begin.x
    delta.y = pos_end.y  - pos_begin.y

    if delta.x>maxDx:
        delta.x -= dim
    if delta.x<(-maxDx):
        delta.x += dim

    if delta.y>maxDy:
        delta.y -= dim
    if delta.y<(-maxDy):
        delta.y += dim

    return delta

"""
int Observation::getCollision(const Point2D &pos) const{
  for (int i = 0; i < (int)positions.size(); i++) {
    if (pos == positions[i])
      return i;
  }
  return -1;
}
"""

def getCollision(allagentPos,targetPos):
    for i in range(len(allagentPos)):
        if allagentPos[i]==targetPos:
            return i
    return -1


def wrapPoint(pos):
    while(pos.x > dim):
        pos.x -= dim
    while(pos.x <0):
        pos.x += dim

    while(pos.y > dim):
        pos.y -= dim
    while(pos.y < 0):
        pos.y += dim

    return pos




def softmax_3(a,b,factor):
    a1 = np.exp(factor*a)
    b1 = np.exp(factor*b)
    return (a1/(a1+b1))

def softmax_array(probs,factor):
    exp_probs = np.exp(np.array(probs)*factor)
    total_sum = np.sum(exp_probs)

    return exp_probs/total_sum
