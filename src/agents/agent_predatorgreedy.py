import numpy as np
from src.agents.common import *



def getGreedyDesiredPosition(obs):
    preyPos = obs.allPos[obs.preyInd]
    myPos = obs.allPos[obs.myInd]
    minDist = 100000
    minPos = defs.Point2D(0,0)
    pos = defs.Point2D(0,0)


    for action in defs.Actions:
        pos = movePosition(preyPos,const.ACTION_TO_MOVES[action])
        col = getCollision(obs.allPos,pos)
        if(col==obs.myInd):
            return preyPos
        elif col<=0:
            dist = getDistanceToPoint(pos,myPos)
            if(dist<minDist):
                minDist = dist
                minPos = pos

    return minPos

def greedyObstacleAvoid(obs,destPos):

    myPos = obs.allPos[obs.myInd]
    preyPos = obs.allPos[obs.preyInd]
    allPos = obs.allPos

    diff = getDifferenceToPoint(myPos,destPos)

    move1 = defs.Point2D(np.sign(diff.x),0)
    move2 = defs.Point2D(0,np.sign(diff.y))

    pos1 = movePosition(myPos,move1)
    pos2 = movePosition(myPos,move2)

    col1 = getCollision(allPos,pos1)
    col2 = getCollision(allPos,pos2)

    pos1Blocked = ((col1>=0) and (col1 != obs.preyInd))
    pos2Blocked = ((col2>=0) and (col2 != obs.preyInd))

    move = defs.Point2D(0,0)

    if(pos1Blocked & pos2Blocked):
        return np.random.choice(defs.Actions)

    elif ((pos1Blocked) & (~pos2Blocked) & (move2 != defs.Actions.NOOP)):
        move = move2

    elif ((pos2Blocked) & (~pos1Blocked) & (move1 != defs.Actions.NOOP)):
        move = move1

    else:
        if(abs(diff.x)>abs(diff.y)):
            move = move1
        else:
            move = move2


    for action in defs.Actions:
        if (move==const.ACTION_TO_MOVES[action]):
            return action

    raise Exception("Should never happen")
    return np.random.choice(defs.Actions)


def step(obs):
    """

    :param allPos: List of all positions
    :param myInd: Index of the current agent aka my index
    :param preyInd: Index of the prey agent aka prey index
    :return: action probabilites
    """

    destPos = getGreedyDesiredPosition(obs)
    action = greedyObstacleAvoid(obs,destPos)

    return const.ACTIONS_TO_ACTIONPROBS[action]


