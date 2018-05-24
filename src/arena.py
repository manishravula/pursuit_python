import numpy as np
# import time
import threading
from copy import deepcopy
from src.utils import pursuit_visualizer as pursuit_vis
import logging
logger = logging.getLogger(__name__)
import src.agents.common as autils
import src.global_defs as defs
import src.global_const as const
import experiments.configuration as config
import time



"""
ITEM should also be an object. with position and capacity.

"""
"""
WORLD:

1) Methods:
    a) Instantiate - start with locations of food, agents and parameters of agents, objects of other agents.
    b) ExternalStep - Recieve a step-next position from the external agent object. 
    c) InternalStep - Recieve a step-next position from the internal agent object.
    d) Check the world for consequences - And enforce them.
    e) If there is a reward mechanism, save it. 

2) MCTS Wrapper:
    a) How does MCTS connect to everything else?
    b) MCTS sub-simulation of the entire world. So each MCTS run utilizes a new WORLD instance to run forward.

"""

"""
REFDOC for ARENA Class:

1) Connections.
        
                                                           ->/   Agent1     \->
                                                          ->/    Agent2      \->   
                                                           /                  \ 
     Variables: Grid_matrix,item_list,agent_list          /                    \    Variables: Pos, parameters, orientation. (The passed objects include this)
     Methods: simulation_nextstep()-Make food disappear   \                    /    Methods: Run next_step. -> Should give the next position. 
                                                           \                  /      
                                                          ->\    Agent3      /->
                                                           ->\   MCTSAgent1 /->
                                                        
2) Each simluation step should also update the info in the grid_matrix to make the pygame animation display run well.
3) arena.update()
    This function should be updating the arena after every object has moved.
    It needs to 
      a) Check agents reaching to pick up any object. 
          if they do, make that object disappear by deleting it off the grid_matrix
          else
          let everything be.
      b) Update visualization - through pygame
4) This function

##THE SIM CLASS HAS TO BE SEPERATE - BECAUSE THE MCTS agent has to play often!

Methods:
    1) init()
    2) MCTS - helpers. 
        current_board()
        players_


"""


class arena():
    def __init__(self,prey_loc,agents_list,visualize):
        self.agents = []
        self.visualize = visualize
        self.terminal = False
        self.init_add_prey(prey_loc)
        self.init_add_agents(agents_list)
        if self.visualize:
            allPos = [self.prey_loc] + [self.build_agentPositionArray()]
            obs = defs.obs(self.build_agentPositionArray(),-1,0)
            self.init_visualization(obs)

        self.center_point = defs.Point2D(0,0)
        self.center_point.x = int(config.DIMENSIONS/2)
        self.center_point.y = int(config.DIMENSIONS/2)


    def init_add_agents(self,agents_list):
        #Add agent objects once they are created.
        #The last agent added is a dummy agent, used for MCTS
        self.agents = agents_list
        self.no_agents = len(self.agents)

    def init_add_prey(self,prey_loc):
        self.prey_loc = defs.Point2D(prey_loc[0],prey_loc[1])

    def init_visualization(self,obs):
        self.visualizer = pursuit_vis.pursuit_visualizer(config.DIMENSIONS,obs)
        self.visualize_thread = threading.Thread(target=self.visualizer.wait_on_event)
        self.visualize_thread.start()


    def build_agentPositionArray(self):
        posarray = []
        for agent in self.agents:
            posarray.append(agent.pos)
        return posarray




    def step(self):
        agent_actions = []
        agent_approvedActions = []
        agent_probs = []

        #retrieve what the agent wants to do
        allPos = [self.prey_loc]+self.build_agentPositionArray()
        preyInd = 0
        for (adx,agent) in enumerate(self.agents):
            #Check what the agent wants to do
            myInd = adx+1
            curr_obs = defs.obs(allPos,myInd,preyInd)
            action_probs = agent.behave(curr_obs)
            agent_probs.append(action_probs)

        #Get actions
        for adx in range(len(self.agents)):
            action = np.random.choice(defs.Actions,p=agent_probs[adx])
            agent_actions.append(action)

        #Resolve collisions
        random_ordering = range(len(self.agents))
        np.random.shuffle(random_ordering)
        for adx in range(len(self.agents)):
            agent_idx = random_ordering[adx]
            agent = self.agents[agent_idx]
            agent_requestAction = agent_actions[agent_idx]
            agent_requestMovement = const.ACTION_TO_MOVES[agent_requestAction]
            agent_requestPosition = autils.movePosition(agent.pos,agent_requestMovement)

            #Check if it collides, move if it doesn't.

            curr_agentPosList = self.build_agentPositionArray()
            all_posList = [self.prey_loc]+curr_agentPosList
            col = autils.getCollision(all_posList,agent_requestPosition)
            if (col>=0):
                agent_requestAction = defs.Actions.NOOP #NOOP
            else:
                agent.behave_act(agent_requestAction)

            agent_approvedActions.append(agent_requestAction)

        if self.visualize:
            self.update_vis()



        self.prey_step()
        if config.SIM_DELAY:
            time.sleep(config.SIM_DELAY/3)

        if self.visualize:
            self.update_vis()

        # self.world_center()

        if self.visualize:
            self.update_vis()

        return agent_actions,agent_probs

    def update_vis(self):
        agent_pos_array = self.build_agentPositionArray()
        all_posarray = [self.prey_loc] + agent_pos_array
        vis_obs = defs.obs(all_posarray,-1,0)
        self.update_event = pursuit_vis.pygame.event.Event(self.visualizer.update_event_type,{'obs': vis_obs})
        pursuit_vis.pygame.event.post(self.update_event)
        # self.visualizer.snapshot(str(time.time()))

    def prey_step(self):
        #Find open location
        validMovements = []
        allagentPos = self.build_agentPositionArray()
        for action in defs.Actions:
            if action==defs.Actions.NOOP:
                break
            else:
                movement = const.ACTION_TO_MOVES[action]
                final_pos = autils.movePosition(self.prey_loc,movement)
                if (autils.getCollision(allagentPos,final_pos))>=0:
                    pass
                else:
                    validMovements.append(movement)
        if len(validMovements)==4:
            #No predator around.
            validMovements_probability = [.125,.125,.125,.125,.5] #Mostly noop
            random_action = np.random.choice(defs.Actions,p=validMovements_probability)
            random_movement = const.ACTION_TO_MOVES[random_action]
            self.prey_loc = autils.movePosition(self.prey_loc,random_movement)
            self.terminal = False

        elif len(validMovements)>0:
            random_movement_idx = np.random.choice(range(len(validMovements)))
            random_movement = validMovements[random_movement_idx]
            self.prey_loc = autils.movePosition(self.prey_loc,random_movement)
            self.terminal = False

        else:
            self.terminal = True

    def world_center(self):
        #center the prey so everything looks easy to debug
        if self.prey_loc != self.center_point:
            movement = self.center_point-self.prey_loc
            self.prey_loc = self.center_point
            for agent in self.agents:
                agent.pos = autils.movePosition(agent.pos,movement)


    # def __getstate__(self):
        # cp = deepcopy
        # dict_state = {}
        # dict_state['grid_matrix'] = cp(self.grid_matrix)
        # dict_state['no_agents'] = cp(self.no_agents)
        # dict_state['no_items'] = cp(self.no_items)
        # return cp(dict_state)

    # def __setstate__(self,state):
    #     self.__dict__.update(state)
    #     self.init_build_itemObjects() #Required when agent uses arena.items to check its surroundings.


    def check_for_termination(self):
        prey_loc = self.prey_loc
        allAgent_pos = self.build_agentPositionArray()
        for movement in const.ACTION_TO_MOVES.values():
            final_pos = autils.movePosition(prey_loc,movement)
            if ~(autils.getCollision(allAgent_pos,final_pos)):
                self.terminal = False
                return False
        self.terminal = True
        return True





