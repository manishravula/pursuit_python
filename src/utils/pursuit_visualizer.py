import numpy as np
import matplotlib.pyplot as plt
import time

import pygame
import pygame.camera
import time
import pdb
import threading

#An implementation of the level-based foraging domain.
#10by10 or 15by15 is the right size to go.

pygame.init()
pygame.mixer.quit()
import experiments.configuration as config
#What this code should expect
"""
Initial parameters.
1) mbym array of 'blocks' in the arena with '0's if empty and 'weight' if food is present.
2) The agents and their properties to display
    a)Their capacity
    b)Their viewcone-radius and angle

Time-varying parameters
1) Location and orientation of each agent.
2) Presence/Absence of food

"""



class pursuit_visualizer():
    def __init__(self,grid_size,obs):
        """
        :param food_matrix: mxm numpy array where mxm is the size of the arena. weight if food is present, zero otherwise.
        :param agents_parameters: nby3 array where n is number of agents and each row is [capacity,radius,viewconeangle] of individual agent.
        """

        # pygame.init()
        # pygame.mixer.quit()
        self.grid_shape = (grid_size,grid_size)

        self.agentPos_list = obs.allPos[1::]
        self.preyPos = obs.allPos[obs.preyInd]

        #Setting up display related parameters.
        self.box_size = 60 #Each of the boxes in the domain is 60px wide
        self.screen = pygame.display.set_mode((self.grid_shape[0]*self.box_size,self.grid_shape[1]*self.box_size))
        self.clock = pygame.time.Clock()
        self.screen.fill((255,255,255))
        self.done=False
        self.screen_memory = self.screen

        #setting up text related parameters.
        pygame.font.init()
        self.font = pygame.font.SysFont("comicsansms", 25)
        # /self.wait_on_event()
        self.update_event_type = pygame.USEREVENT+1
        # self.wait_on_event()
        return

    def draw(self,obs):
        self.draw_arena()
        self.text_overlay()
        self.draw_agents(obs.allPos[1::])
        self.draw_prey(obs.allPos[obs.preyInd])

    def visualize(self,obs):
        self.agentPos_list = obs.allPos[1::]
        self.preyPos = obs.allPos[obs.preyInd]
        self.screen.fill((255,255,255))
        self.draw(obs)
        # time.sleep(1)



    def draw_arena(self):
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                x = j*self.box_size
                y = i*self.box_size
                color = (0,128,255) #Blue borders.
                pygame.draw.rect(self.screen,color,pygame.Rect(x,y,self.box_size,self.box_size),2)
        pygame.display.flip()

    def text_overlay(self):
        """
        The function looks at all non-zero weight values and overalys text of weight-values accordingly.
        :return:
        """
        for id,loc in enumerate(self.agentPos_list):
            [x,y] = [loc.x*self.box_size,(config.DIMENSIONS-1-loc.y)*self.box_size]
            item_weight = id
            text = self.font.render(str('{:02.2f}').format(item_weight),True,(0,128,0))
            self.screen.blit(text,(x+self.box_size//2 - text.get_width()//2, y+self.box_size//2 -text.get_height()//2 ))

        pygame.display.flip()

    def draw_prey(self,prey_loc):
        agentspos_list = [prey_loc]
        positions = np.array([[agentpos.x,config.DIMENSIONS-1-agentpos.y] for agentpos in agentspos_list]) #- np.array([1,1]) #flipping lr because the xy cordinate axes and numpy are diff.
        no_agents = len(positions)
        #get center pixel loc of each box involved.
        positions_pixels = np.array((positions+.5)*self.box_size).astype('int')

        for (pos,i) in zip(positions_pixels,range(no_agents)):
            #circular shape
            pygame.draw.circle(self.screen,(255,0,0),[pos[0],pos[1]],int(self.box_size/2),2)
            #text displaying weight
            text = self.font.render(str('{:02.2f}'.format(i)), True, (255,0,0))

        pygame.display.flip()

        return




    def draw_agents(self,agentsPos_list):
        """
        :param positions: Expects position in terms of in which box the agent is in the grid.
        :param orientations: Expects orientation in terms of the radian angle in anti-clockwise direction with respect to x-axis
        :return: Nothing, just draws.
        """

        positions = np.array([[agentpos.x,config.DIMENSIONS-1-agentpos.y] for agentpos in agentsPos_list]) #- np.array([1,1]) #flipping lr because the xy cordinate axes and numpy are diff.
        no_agents = len(positions)
        #get center pixel loc of each box involved.
        positions_pixels = np.array((positions+.5)*self.box_size).astype('int')

        for (pos,i) in zip(positions_pixels,range(no_agents)):
            #circular shape
            pygame.draw.circle(self.screen,(0,255,0),[pos[0],pos[1]],int(self.box_size/2),2)
            #text displaying weight
            text = self.font.render(str('{:02.2f}'.format(i)), True, (65,105,225))
        pygame.display.flip()

        return


    def wait_on_event(self):
        while not self.done:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                self.done = True
                pygame.display.quit()
                pygame.quit()
                break
            elif event.type == self.update_event_type:
                self.visualize(event.obs)


    def snapshot(self,name):

        # grab first frame
        pygame.image.save(self.screen, name+str('.png'))
        return

#
# if def _nam
#
# dummy_params = np.random.random((2,3))
# dummy_params[:,1]*=500
# dummy_params[:,2]*=2
# dummy_food = np.zeros((10,10))
# dummy_food [1,1] = .09
# dummy_food [1,2] = .1
# dummy_pos = np.array([[5,3],[4,4]])
# orientations = np.array([4,2]).astype('float')
# capacities = np.array([4,4])
# a = LVDvisualizer(dummy_food,dummy_params)
# vis_thread = threading.Thread(target=a.wait_on_event)
# vis_thread.start()
# time.sleep(1)
#
# def vis():
#     print('lol')
#     for i in range(10):
#         print('Sending')
#         dummy_pos = np.array([[5,3],[4,4]])
#         dummy_pos +=np.random.randint(-2,2,(2,2))
#         orientations = np.array([4,2]).astype('float')
#         orientations += (np.random.random(2)*4 - 2)
#         event = pygame.event.Event(a.update_event_type,{'food_matrix': dummy_food,'agents_positions': dummy_pos, 'agents_orientations': orientations})
#         pygame.event.post(event)
#         time.sleep(1)
#
# vis()
# # t2 = threading.Thread(target=vis)
# # t2.start()






