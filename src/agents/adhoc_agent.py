import numpy as np
import src.agents.agent_predatorgreedy as a_predGreedy
import src.agents.agent_predatorgreedyprobabilistic as a_predGreedyProb
import src.agents.agent_PredatorProbabilisticDestinations as a_predProbDest
import src.agents.agent_PredatorTeamMateAware as a_predTeamAware

from src.agents.common import *
from experiments import configuration as config

from src.agents import agent_factory as afactory




def estimate_type(trajectory):
    dummyagentlist = []
    for tp in range(4):
        dummyagentlist.append(afactory.agent(tp,(0,0)))
    likelihoods = []
    evidence = [1,1,1,1]
    for step_obs in trajectory.listOfObservations:
        llist = []
        for tp in range(4):
            llist.append(dummyagentlist[tp].behave(step_obs.obs)[step_obs.action-1])
            evidence[tp]*=llist[tp]
        curr_type = np.argmax(evidence)
        print(curr_type)
    return curr_type




