import src.arena as arena
import src.agents.agent_factory as agent_factory
import time
import experiments.configuration as config


agent1 = agent_factory.agent(0,(12,2))
agent2 = agent_factory.agent(0,(13,3))
agent3 = agent_factory.agent(0,(4,14))
agent4 = agent_factory.agent(0,(3,8))
agent5 = agent_factory.agent(1,(7,7))
agent6 = agent_factory.agent(1,(8,8))


agents_list = [agent1,agent2,agent3,agent4]#,agent5,agent6]

a = arena.arena([5,5],agents_list,True)
i=0
while not a.terminal:
    a.step()
    time.sleep(config.SIM_DELAY)
    print(i)
    i+=1