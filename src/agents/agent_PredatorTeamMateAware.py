import numpy as np
from src.agents.common import *
import src.astar.Algorithms as astar
import src.astar.Graph as astar_graph


NUM_PREDATORS = 4
NUM_DESTS = 4
astar_grid = astar_graph.SquareGrid(config.DIMENSIONS,config.DIMENSIONS)

def assignTeammateAwareDesiredDests(obs, stopAfterAssigningCurrentPred, moveOntoPreyIfAtDest,distFactor):
    myInd = obs.myInd
    preyInd = obs.preyInd
    allPos = obs.allPos
    preyPos = allPos[preyInd]
    myPos = allPos[myInd]


    assert (preyInd==0)

    #Check how far each predator is to each surrounding spot
    distances = np.empty((NUM_PREDATORS,NUM_DESTS),dtype='int')

    possibleDests = [defs.Point2D(0,0) for i in range(NUM_DESTS)]

    minDists = np.empty(NUM_PREDATORS,dtype='int')
    minInds = np.empty(NUM_PREDATORS,dtype='int')

    for pred in range(NUM_PREDATORS):
        minDists[pred] = 999999
        minInds[pred] = 0


    dest = defs.Point2D(0,0)

    for destInd in range(NUM_DESTS):
        mp = const.ACTION_TO_MOVES[destInd+1]
        mp.x *= distFactor
        mp.y *= distFactor
        dest = movePosition(preyPos,mp)
        possibleDests[destInd] = dest

        for pred in range(NUM_PREDATORS):
            distances[pred][destInd] = getDistanceToPoint(allPos[pred+1],dest)
            if(distances[pred][destInd] < minDists[pred]):
                minDists[pred] = distances[pred][destInd]
                minInds[pred] = destInd

    maxDistPred = 0
    chosenDest = 0
    for numUnassignedPreds in range(NUM_PREDATORS,0,-1):
        maxDist = -1
        for pred in range(NUM_PREDATORS):
             if(minDists[pred]>maxDist):
                maxDist=minDists[pred]
                maxDistPred=pred


        chosenDest = minInds[maxDistPred]
        dests = [defs.Point2D(0,0) for _ in range(NUM_PREDATORS)]
        dests[maxDistPred] = possibleDests[chosenDest]


        #if the predator is already at that position, move on to the prey
        if(moveOntoPreyIfAtDest & (dests[maxDistPred] == allPos[maxDistPred+1])):
            dests[maxDistPred] = preyPos
        if(stopAfterAssigningCurrentPred & (maxDistPred+1 == myInd)):
            return dests

        #make it clear this predator has chosen
        minDists[maxDistPred] = -1

        #remove this option for other predators
        for pred in range(NUM_PREDATORS):
            if(minDists[pred]<0):
                continue
            distances[pred][chosenDest] = 999999
            if (minInds[pred]==chosenDest):
                minDists[pred] = 999999
                for neighbor in range(NUM_DESTS):
                    if distances[pred][neighbor]  < minDists[pred]:
                        minDists[pred] = distances[pred][neighbor]
                        minInds[pred] = neighbor
    return dests



def getTeammateAwareDesiredPosition(obs):
    dests = assignTeammateAwareDesiredDests(obs,True,True,1)
    return dests[obs.myInd - 1]

def step(obs):
    dest = getTeammateAwareDesiredPosition(obs)
    walls = []
    for i in range(len(obs.allPos)):
        if i!=obs.myInd and i!=obs.preyInd:
            walls.append((obs.allPos[i].x,obs.allPos[i].y))

    astar_grid.walls = walls
    myPos = obs.allPos[obs.myInd]
    myPos_tuple = (myPos.x,myPos.y)
    dest_tuple = (dest.x,dest.y)
    route,cost = astar.a_star_search(astar_grid,myPos_tuple,dest_tuple)

    if dest_tuple in route.keys():
        path = astar.reconstruct_path(route,myPos_tuple,dest_tuple)
        first_block = path[1]
        first_block_point = defs.Point2D(first_block[0],first_block[1])
        diff = getDifferenceToPoint(obs.allPos[obs.myInd],first_block_point)
        return const.ACTIONS_TO_ACTIONPROBS[const.MOVES_TO_ACTIONS[str(diff)]]

    else:
        action = np.random.choice(defs.Actions)
        return const.ACTIONS_TO_ACTIONPROBS[action]







