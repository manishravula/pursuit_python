from src import global_defs as defs
import pickle
from src.agents import agent_factory as afactory
import numpy as np
from src import champ as champ



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
    return curr_type

def estimate_individual_type(trajectory,tp):
    dummyagent = afactory.agent(tp,(0,0))
    likelihoods = []
    evidence = 1
    for step_obs in trajectory.listOfObservations:
        ll = 1
        ll = dummyagent.behave(step_obs.obs)[step_obs.action-1]
        evidence *= ll
    return evidence




class champ_middleware():
    def __init__(self,min_length,mean_length,std_length,max_particles,resamp_particles):
        self.observations = []
        self.fitters = [self.fitter_type1,self.fitter_type2,self.fitter_type3,self.fitter_type4]
        config = champ.Config(self.fitters, length_min=min_length, length_mean=mean_length, length_sigma=std_length, max_particles=max_particles,
                        resamp_particles=resamp_particles)
        self.champ_instance = champ.Champ(config)


    def add_observation(self,obs):
        self.observations.append(obs)

    def observe(self,adhoc_obs):
        self.champ_instance.observe(len(self.observations)+1,adhoc_obs.action)
        self.add_observation(adhoc_obs)
        return self.champ_instance.backtrack(len(self.observations)-1)

    def fitter_type1(self,i,j):
        if i<0 or j>(len(self.observations)+1) or i>j:
            raise Exception("fitting between {} and {} incompatible when number of acquisitions have been {}".format(i,j,len(self.observations)))

        curr_trajectory = defs.trajectory(0,self.observations[i:j])
        res1 = estimate_individual_type(curr_trajectory,0)
        return res1,0.1

    def fitter_type2(self,i,j):
        if i<0 or j>(len(self.observations)+1) or i>j:
            raise Exception("fitting between {} and {} incompatible when number of acquisitions have been {}".format(i,j,len(self.observations)))

        curr_trajectory = defs.trajectory(1,self.observations[i:j])
        res2 = estimate_individual_type(curr_trajectory,1)
        return res2,0.1

    def fitter_type3(self,i,j):
        if i<0 or j>(len(self.observations)+1) or i>j:
            raise Exception("fitting between {} and {} incompatible when number of acquisitions have been {}".format(i,j,len(self.observations)))

        curr_trajectory = defs.trajectory(2,self.observations[i:j])
        res3 = estimate_individual_type(curr_trajectory,2)
        return res3,0.1

    def fitter_type4(self,i,j):
        if i<0 or j>(len(self.observations)+1) or i>j:
            raise Exception("fitting between {} and {} incompatible when number of acquisitions have been {}".format(i,j,len(self.observations)))

        curr_trajectory = defs.trajectory(3,self.observations[i:j])
        res4 = estimate_individual_type(curr_trajectory,3)
        return res4,0.1




if __name__=='__main__':
    with open('traj','rb') as handle:
        traj_list = pickle.load(handle)
    for traj in traj_list:
        res_list = []
        champ_observer = champ_middleware()
        for adhoc_obs in traj:
            res_list.append(champ_observer.observe(adhoc_obs))
            print(res_list[-1])




