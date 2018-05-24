import numpy as np
import src.agents.agent_predatorgreedy as a_predGreedy
import src.agents.agent_predatorgreedyprobabilistic as a_predGreedyProb
import src.agents.agent_PredatorProbabilisticDestinations as a_predProbDest
import src.agents.agent_PredatorTeamMateAware as a_predTeamAware

from src.agents.common import *




class agent():
    def __init__(self,type,init_pos):
        self.type = type
        self.pos = defs.Point2D(0,0)
        self.pos.x = init_pos[0]
        self.pos.y = init_pos[1]

    def behave(self,obs):

        if self.type == 0:
            action_probs = a_predGreedy.step(obs)
        elif self.type == 1:
            action_probs = a_predGreedyProb.step(obs)
        elif self.type == 2:
            action_probs = a_predProbDest.step(obs)
        elif self.type ==3:
            action_probs = a_predTeamAware.step(obs)
        else:
            raise Exception("Unknown agent requested")

        return action_probs

    def behave_act(self,action):
        self.pos = movePosition(self.pos,const.ACTION_TO_MOVES[action])









