import numpy as np
from src.agents.common import *
from src.agents import agent_predatorgreedy as apgreedy

blockedPenalty = 3
dimensionFactor = 2
directionFactor = -2


def step(obs):
    """
    Function to step through iterations
    :param allPos: All positions
    :param myInd: Index of the agent
    :param preyInd: Index of the prey
    :return:
    """

    allPos = obs.allPos
    myInd = obs.myInd
    preyInd = obs.preyInd

    myPos = allPos[myInd]
    preyPos = allPos[preyInd]

    desiredPosition = apgreedy.getGreedyDesiredPosition(obs)

    dists = [defs.Point2D(0,0),defs.Point2D(0,0)]
    minDists = defs.Point2D(0,0)

    dists[0] = wrapPoint(desiredPosition-myPos)
    dists[1] = wrapPoint(myPos-desiredPosition)

    minDists.x = min(dists[0].x,dists[1].x)
    minDists.y = min(dists[0].y,dists[1].y)

    move = const.Point2D(0,0)
    pos = const.Point2D(0,0)

    collision = False

    for action in defs.Actions:
        move = const.ACTION_TO_MOVES[action]
        pos = movePosition(myPos,move)

        collision = getCollision(allPos,pos)

        if((collision>=0) & (collision != preyInd)):
            if (move.x>0):
                dists[0].x += blockedPenalty
            elif(move.x<0):
                dists[1].x += blockedPenalty
            elif(move.y>0):
                dists[0].y += blockedPenalty
            elif(move.y<0):
                dists[1].y += blockedPenalty


    probMinX = softmax_3(minDists.x,minDists.y,dimensionFactor)
    prob = softmax_3(dists[0].x,dists[1].x,directionFactor)
    probs = [probMinX*prob, probMinX*(1-prob),0,0,0]
    prob = softmax_3(dists[0].y,dists[1].y,directionFactor)
    probs[2] = (1-probMinX)*prob
    probs[3] = (1-probMinX)*(1.0-prob)
    return probs
