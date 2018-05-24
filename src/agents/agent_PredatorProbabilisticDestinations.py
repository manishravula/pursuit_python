import numpy as np
from src.agents.common import *



distanceFromPreyFactor = -1
distanceFromCurrentFactor = -1

class agent_ppd():
    def __init__(self):
        self.actionProbs = np.empty(5,dtype=float)
        return

    def setDistanceProbs(self,distanceToPrey):
        self.distances = []
        maxDist = dim/2
        for i in range(1,maxDist+1):
            self.distances.append(i)
        self.distanceProbs = softmax_array(self.distances,distanceFromPreyFactor)

    def setDestinationsForDistance(self,obs,dist):
        preyPos  = obs.allPos[obs.preyInd]
        self.destinations = []
        diff = defs.Point2D(dist,0)
        change = defs.Point2D(-1,1)

        while(diff.x > (-dist)):
            self.destinations.append(movePosition(preyPos,diff))
            if diff.y == dist:
                change.y *= -1
            diff += change

        change.x *= -1
        while(diff.x < dist):
            self.destinations.append(movePosition(preyPos,diff))
            if (diff.y == -dist):
                change.y *= -1
            diff += change

    def evaluateDestinations(self,obs,distanceProb):
        diff = defs.Point2D(0,0)
        chosenMove = defs.Point2D(0,0)
        nextPos = defs.Point2D(0,0)
        moves = []
        destDists = []
        destProbs =[]

        for i in range(len(self.destinations)):
            if(getCollision(obs.allPos,self.destinations[i])):
                pass
            diff = getDifferenceToPoint(obs.allPos[obs.myInd],self.destinations[i])
            if (abs(diff.x) > abs(diff.y)):
                chosenMove.x = np.sign(diff.x)
                chosenMove.y = 0
            else:
                chosenMove.x = 0
                chosenMove.y = np.sign(diff.y)
            nextPos = movePosition(obs.allPos[obs.myInd],chosenMove)
            if nextPos==obs.allPos[obs.myInd]:
                assert nextPos != obs.allPos[obs.myInd]
            if getCollision(obs.allPos,nextPos) >= 0:
                pass
            moves.append(const.MOVES_TO_ACTIONS[str(chosenMove)])
            destDists.append(diff.manhattan_dist())

        if (len(destDists)==0):
            #No valid destinations, just move randomly
            for i in range(5):
                self.actionProbs[i] += distanceProb/5
        else:
            destProbs = softmax_array(destDists,distanceFromCurrentFactor)
            for i in range(len(destProbs)):
                self.actionProbs[moves[i]-1] += distanceProb * destProbs[i]

    def step(self,obs):
        self.actionProbs*=0
        myPos = obs.allPos[obs.myInd]
        preyPos = obs.allPos[obs.preyInd]

        distanceToPrey = getDistanceToPoint(myPos,preyPos)
        if(distanceToPrey==1):
            move = getDistanceToPoint(myPos,preyPos)
            action = const.MOVES_TO_ACTIONS[str(move)]
            action_probs= const.ACTIONS_TO_ACTIONPROBS[action]
            return action_probs

        self.setDistanceProbs(distanceToPrey)
        for i in range(len(self.distances)):
            self.setDestinationsForDistance(obs,self.distances[i])
            self.evaluateDestinations(obs,self.distanceProbs[i])
        return np.copy(self.actionProbs)



a = agent_ppd()
def step(obs):
    return a.step(obs)






