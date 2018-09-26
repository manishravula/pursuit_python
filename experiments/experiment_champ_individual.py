import src.arena as arena
import src.agents.agent_factory as agent_factory
import time
import experiments.configuration as config
import numpy as np
import pdb
from collections import namedtuple
from src import global_defs as defs
from src.agents import adhoc_changepoint_agent as achapagent
from src import global_const as const
import random

# agent1 = agent_factory.agent(0,(12,2))
# agent2 = agent_factory.agent(0,(13,3))
# agent3 = agent_factory.agent(0,(4,14))
# agent4 = agent_factory.agent(0,(3,8))
# agent5 = agent_factory.agent(1,(7,7))
# agent6 = agent_factory.agent(1,(8,8))


# agents_list = [agent1,agent2,agent3,agent4]#,agent5,agent6]


# a = arena.arena([5,5],agents_list,True)
# for agent in a.agents:
#     agent.tp = 2
import numpy as np
from enum import IntEnum
from recordclass import recordclass
from collections import namedtuple

_point2d = recordclass('Point','x y')
obs = recordclass('Observation','allPos myInd preyInd')
trajectory = namedtuple('trajectory','type listOfObservations')
step_observation = namedtuple('ObservationStep','timestep obs action ap')

class Point2D(_point2d):

    def __add__(self,other):
        x = self.x+other.x
        y = self.x+other.y
        return Point2D(x,y)

    def __sub__(self, other):
        "self-other"
        x = self.x - other.x
        y = self.y - other.y
        return Point2D(x,y)

    def __str__(self):
        return str(self.x)+str(self.y)

    def __hash__(self):
        return self.__str__()

    def __eq__(self, other):
        if ((self.x==other.x)and(self.y==other.y)):
            return True
        else:
            return False

    def as_array(self):
        return np.array([self.x,self.y])

    def as_tuple(self):
        return (self.x,self.y)

    def manhattan_dist(self):
        return abs(self.x)+abs(self.y)

i=0

tot_times = []

tracking_agent_id = 0

traj_list = []
cp_traj_list = []

set_of_types = set([0,1,2,3])


for lm in range(2):
    times = []
    for i in range(4):
        curr_time = []

        for j in range(3):
            idx=0
            list_of_observations = []

            a = arena.arena.create_rndAgents_rndPrey(4,False)
            for agent in a.agents:
                agent.tp=(i+3)%4
            pre_type = a.agents[tracking_agent_id].tp

            while(not a.terminal and idx<50):
                tracking_currObs = a.create_observation_for_agent(tracking_agent_id)

                actions,aps = a.step()

                idx+=1

                curr_obs = defs.adhoc_observation(idx,tracking_currObs,actions[tracking_agent_id],aps[tracking_agent_id])
                list_of_observations.append(curr_obs)

                if idx==55:
                    print("Adding changepoint")
                    set_of_possible_new_types = set_of_types.copy()
                    set_of_possible_new_types.remove(pre_type)
                    post_type = random.sample(set_of_possible_new_types,1)
                    a.agents[tracking_agent_id].tp = post_type[0]

                print("set {} iter {} exper {} tp {}".format(lm,idx,j,i))

            if a.terminal:
                print("Simulation ended")
                curr_time.append(idx)

            curr_traj = defs.trajectory(a.agents[tracking_agent_id].tp,list_of_observations)
            if idx>56:
                cp_curr_traj = defs.cp_trajectory(pre_type,post_type,25,list_of_observations)
                cp_traj_list.append(cp_curr_traj)
            traj_list.append(curr_traj)

        times.append(curr_time)

    times = np.array(times)
    # print(np.mean(times,axis=1))
    tot_times.append(times)


# print(np.mean(tot_times,axis=0))
import pickle
import src.agents.adhoc_agent as sadhoc

# with open("1000x100x2_dump",'wb') as handle:
#     pickle.dump(traj_list,handle)

#sadhoc.estimate_type(traj_list[0])

#Create dictionary of observations
bag_of_obs = {}
bag_of_obs[0] = []
bag_of_obs[1] = []
bag_of_obs[2] = []
bag_of_obs[3] = []

for traj in traj_list:
    bag_of_obs[traj.type]+=traj.listOfObservations


summary = []
sub_res = namedtuple('summary_elem','popsize, iter, truetp, estimatedtype')
#Experiment to see if random collections of these observations shall result in proper estimation
for i in range(4):
    #For each of type

    for j in range(2,50):
        for k in range(10):
            #for varying set sizes from 0 to 100
            curr_list_of_observations = random.sample(bag_of_obs[i],j)
            curr_traj = defs.trajectory(i,curr_list_of_observations)
            estimated_type = achapagent.estimate_type(curr_traj)
            curr_res = sub_res(j,k,i,estimated_type)
            print("Pop size: {} Iter {} True Type {} Estimated Type {}".format(j,k,i,estimated_type))
            summary.append(curr_res)

#Summary statistics of estimation policy
summary_array = np.array([list(ele) for ele in summary])

def sumstats_matrix(summary_array,sample_size):
    """
    Returns the 4x4 matrix of with rows as true types and cols as estimated types and values as the number of times
    the estimation relation occured
    :param summary_array: list of summaries as generated by the above eperiment Pop size: {} Iter {} True Type {} Estimated Type {}
    :param sample_size: The sample size of the experiments to consider this. this is equivalent to the sum of elements in a row in the result
    :return:
    """
    sub_summary_array = summary_array[np.where(summary_array[:,0]==sample_size)]
    results = []
    for tp in range(4):
        curr_res = []
        for tp2 in range(4):
            curr_res.append(np.sum(np.logical_and(sub_summary_array[:,2]==tp,sub_summary_array[:,3]==tp2)))
        results.append(curr_res)
    return np.array(results)


def sumstats_matrix_aggregate(summary_array, sample_size):
    """
    Returns the 4x4 matrix of with rows as true types and cols as estimated types and values as the number of times
    the estimation relation occured,
    :param summary_array: list of summaries as generated by the above eperiment Pop size: {} Iter {} True Type {} Estimated Type {}
    :param sample_size: The sample size of the experiments to consider lesser than this. this is equivalent to the sum of elements in a row in the result
    :return:
    """
    sub_summary_array = summary_array[np.where(summary_array[:, 0] <= sample_size)]
    results = []
    for tp in range(4):
        curr_res = []
        for tp2 in range(4):
            curr_res.append(np.sum(np.logical_and(sub_summary_array[:, 2] == tp, sub_summary_array[:, 3] == tp2)))
        results.append(curr_res)
    return np.array(results)


def sumstats_matrix_range(summary_array, sample_size_low,sample_size_high):
    """
    Returns the 4x4 matrix of with rows as true types and cols as estimated types and values as the number of times
    the estimation relation occured,
    :param summary_array: list of summaries as generated by the above eperiment Pop size: {} Iter {} True Type {} Estimated Type {}
    :param sample_size: The sample size of the experiments to consider lesser than this. this is equivalent to the sum of elements in a row in the result
    :return:
    """
    sub_summary_array = summary_array[np.logical_and(summary_array[:, 0] <= sample_size_high,summary_array[:,0]>=sample_size_low)]
    results = []
    for tp in range(4):
        curr_res = []
        for tp2 in range(4):
            curr_res.append(np.sum(np.logical_and(sub_summary_array[:, 2] == tp, sub_summary_array[:, 3] == tp2)))
        results.append(curr_res)
    return np.array(results)


# print("Summary matrix for 10")
# print(sumstats_matrix_aggregate(summary_array,10))
#
#
# print("summary matrix for 10-20")
# print(sumstats_matrix_range(summary_array,10,20))
#
# print("summary matrix for 20-30")
# print(sumstats_matrix_range(summary_array,20,30))

# print("summary matrix for 30-40")
# print(sumstats_matrix_range(summary_array,30,40))
#
# while(1):
#     ijk=0


#CHAMP STUFF BEGIN

#Create trajectories with artificial changepoints
n_trajectories_per_set=20
traj_length = 50
changepoint_location = range(20,30)


traj_list = []
for cp_loc in changepoint_location:
    pre_type_list = []
    for pre_type in range(4):
        for post_type in range(4):
            for j in range(n_trajectories_per_set):
                pre_part = random.choice(bag_of_obs[pre_type],cp_loc)
                post_part = random.choice(bag_of_obs[post_type],traj_length-cp_loc)
                post_type_list = pre_part+post_part
        pre_type_list.append(post_type_list)
    traj_list.append(pre_type_list)




res_list =[]
for traj_set in traj_list[0]:
    for pre_tp_set in traj_set:
        pre_tp_res = []
        for post_tp_set in pre_tp_set:
            post_tp_res = []
            for traj in post_tp_set:
                curr_traj_res = []
                champ_tool = achapagent.champ_middleware(18,20,1,50,40)
                for obs in traj:
                    res = champ_tool.observe(obs)
                    curr_traj_res.append(res)
            post_tp_res.append(curr_traj_res)
        pre_tp_res.append(post_tp_res)
    res_list.append(pre_tp_res)








# for traj in cp_traj_list:
#     res_list = []
#     print("Processing trajectory with length {}-------------------------------------".format(len(traj.listOfObservations)))
#     for adhoc_obs in traj.listOfObservations:
#         res_list.append(champ_observer.observe(adhoc_obs))
#         print(res_list[-1])
#     print(res_list[-1])
#     print("Finished processing =---------------------------------------")
#     print(traj.from_type,traj.to_type)
# CHAMP STUFF END


