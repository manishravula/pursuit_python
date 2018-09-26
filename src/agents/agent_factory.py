import numpy as np
import src.agents.agent_predatorgreedy as a_predGreedy
import src.agents.agent_predatorgreedyprobabilistic as a_predGreedyProb
import src.agents.agent_PredatorProbabilisticDestinations as a_predProbDest
import src.agents.agent_PredatorTeamMateAware as a_predTeamAware

from src.agents.common import *
from experiments import configuration as config



class agent():
    def __init__(self,tp,init_pos):
        self.tp = tp
        self.pos = defs.Point2D(0,0)
        self.pos.x = init_pos[0]
        self.pos.y = init_pos[1]

    def behave(self,obs):

        if self.tp == 0:
            action_probs = a_predGreedy.step(obs)
        elif self.tp == 1:
            action_probs = a_predGreedyProb.step(obs)
        elif self.tp == 2:
            action_probs = a_predProbDest.step(obs)
        elif self.tp ==3:
            action_probs = a_predTeamAware.step(obs)
        else:
            raise Exception("Unknown agent requested")

        return self.probability_renormalize(action_probs,0.001)

    def behave_act(self,action):
        self.pos = movePosition(self.pos,const.ACTION_TO_MOVES[action])

    def probability_renormalize(self,probs,factor):
        probs = np.array(probs)+factor
        probs = probs/np.sum(probs)
        return probs

def gen_agents_random(num_agents):
    random_types = np.random.randint(0,config.NO_TYPES,num_agents)
    random_locs_cords = np.random.randint(0,config.DIMENSIONS, num_agents*100)

    random_locs_x = np.random.choice(random_locs_cords,num_agents,replace=False)
    random_locs_y = np.random.choice(random_locs_cords,num_agents,replace=False)

    random_locs = [[i,j] for (i,j) in zip(random_locs_x,random_locs_y)]

    agents_random = []
    for tp,loc in zip(random_types,random_locs):
        agents_random.append(agent(tp,tuple(loc)))

    return agents_random














