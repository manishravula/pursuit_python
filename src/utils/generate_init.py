import copy
import logging
import os

import numpy as np

import experiments.configuration as config

logger = logging.getLogger(__name__)


def generate_arena_matrix(size, n_items, ret_numitems=True):
    """

    :param ret_numitems:
    :param size: The size (in nxn) of the grid
    :param n_items: Approx. number of items. Rarely, after creation, some agents might fall in tha place of items
                    decreasing the total number of items.
    :return: the grid_matrix without the agents' positions being nullified.
    """
    size_pre_pad = size - 2
    grid_matrix = np.random.random((size_pre_pad,size_pre_pad))
    n_ele_prePad = size_pre_pad*size_pre_pad
    g = grid_matrix.flatten()
    g[[np.random.choice(np.arange(n_ele_prePad), n_ele_prePad-n_items, replace=False)]] = 0
    grid_matrix = g.reshape((size_pre_pad, size_pre_pad))
    grid_matrix = np.lib.pad(grid_matrix, (1, 1), 'constant', constant_values=(0, 0))
    grid_matrix /= 2.0
    g2 = copy.deepcopy(grid_matrix)

    final_numitems = np.sum(grid_matrix!=0)

    file_path = '../data/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np.save(open('../data/grid.npy','w+'),grid_matrix)

    if ret_numitems:
        return g2, final_numitems
    else:
        return g2


def generate_agents(n_agents,arena,from_save=False):
    """
    Generate agent locations. The only constraint would be in assigning capacities.
    To randomize as much as possible.
    :param n_agents: number of agents.
    :param arena: nxn matrix.
    :return:
    """
    grid_matrix = arena.grid_matrix

    if not from_save:
        #Capacity vectors.
        max_capacity = np.max(grid_matrix)
        min_capacity = np.max([config.COOPERATION_INDEX*np.min(grid_matrix),.1])
        capacity_array = np.random.random(n_agents)*(max_capacity-min_capacity) + min_capacity

        #View_angles
        max_vangle = .9
        min_vangle = .15
        vangle_array = np.random.random(n_agents)*(max_vangle-min_vangle) + min_vangle

        #View_radius
        max_vradius = .9
        min_vradius = .2
        vradius_array = np.random.random(n_agents)*(max_vradius-min_vradius) + min_vangle

        #types
        tp_array = np.random.randint(0,config.NO_TYPES-1,n_agents)

        positions_array = np.random.randint(0,np.shape(arena.grid_matrix)[0],(n_agents,2))

        agent_params = [tp_array,capacity_array,vangle_array,vradius_array,positions_array[:,0],positions_array[:,1]]

        file_path = '../data/'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        np.save(open('../data/agentconfig.npy', 'w+'), agent_params)

    else:
        (tp_array,capacity_array,vangle_array,vradius_array,positions_array_1,positions_array_2) =  np.load(open("../data/agentconfig.npy",'r'))
        positions_array = np.vstack((positions_array_1,positions_array_2)).T.astype('int')
        tp_array = tp_array.astype('int')


    agent_list = []
    for i in range(n_agents):
        cap  = capacity_array[i]
        vradius = vradius_array[i]
        vangle = vangle_array[i]
        pos = positions_array[i]
        tp = tp_array[i]
        logger.debug('creating agent with capacity {} vradius {} vangle {} type {} loc {}'.format(cap,vradius,vangle,tp,pos))
        agent_list.append(config.AGENT_CURR(cap,vradius,vangle,tp,pos,arena))

    return agent_list

def generate_agents_tosave(n_agents,grid_matrix):
    # Capacity vectors.
    max_capacity = np.max(grid_matrix)
    min_capacity = np.max([config.COOPERATION_INDEX * np.min(grid_matrix), .1])
    capacity_array = np.random.random(n_agents) * (max_capacity - min_capacity) + min_capacity

    # View_angles
    max_vangle = .9
    min_vangle = .15
    vangle_array = np.random.random(n_agents) * (max_vangle - min_vangle) + min_vangle

    # View_radius
    max_vradius = .9
    min_vradius = .2
    vradius_array = np.random.random(n_agents) * (max_vradius - min_vradius) + min_vangle

    # types
    tp_array = np.random.randint(0, config.NO_TYPES - 1, n_agents)

    positions_array = np.random.randint(0, np.shape(grid_matrix)[0], (n_agents, 2))
    agent_params = [tp_array, capacity_array, vangle_array, vradius_array, positions_array[:, 0], positions_array[:, 1]]

    return agent_params


def generate_agents_fromfile(agent_filename,n_agents,arena):
    """
    Generate agent locations. The only constraint would be in assigning capacities.
    To randomize as much as possible.
    :param n_agents: number of agents.
    :param arena: nxn matrix.
    :return:
    """
    grid_matrix = arena.grid_matrix

    (tp_array,capacity_array,vangle_array,vradius_array,positions_array_1,positions_array_2) =  np.load(open(agent_filename,'r'))
    positions_array = np.vstack((positions_array_1,positions_array_2)).T.astype('int')
    tp_array = tp_array.astype('int')

    agent_list = []
    for i in range(n_agents):
        cap  = capacity_array[i]
        vradius = vradius_array[i]
        vangle = vangle_array[i]
        pos = positions_array[i]
        tp = tp_array[i]
        logger.debug('creating agent with capacity {} vradius {} vangle {} type {} loc {}'.format(cap,vradius,vangle,tp,pos))
        agent_list.append(config.AGENT_CURR(cap,vradius,vangle,tp,pos,arena))

    return agent_list

def generate_all(size,n_items,n_agents):
    grid_matrix, final_numitems = generate_arena_matrix(size, n_items)
    logger.info("Initializing new configuration")
    assert np.shape(grid_matrix)[0]==size, "Grid_matrix not of size {}".format(size)
    logger.debug('creating arena object with grid_matrix {}'.format(grid_matrix))
    arena = config.ARENA_CURR(grid_matrix,visualize=config.VISUALIZE_SIM)
    logger.debug('creating {} agents'.format(n_agents))
    agents_list = generate_agents(n_agents,arena)
    arena.init_add_agents(agents_list)
    logger.debug('Arena created with objects added and new grid_matrix {}'.format(arena.grid_matrix))

    logger.info("Finished creating arena and agents. One item consumed due to overlap. Current item count is {}".format(arena.no_items))


    return arena, agents_list

def generate_reload(n_agents):
    grid_matrix = np.load(open('../data/grid.npy','r'))

    logger.info("Initializing old configuration for grid and agents")
    logger.debug('creating arena object with grid_matrix {}'.format(grid_matrix))
    arena = config.ARENA_CURR(grid_matrix, visualize=config.VISUALIZE_SIM)
    # logger.debug('creating {} agents'.format(n_agents))
    agents_list = generate_agents(n_agents, arena, True)
    arena.init_add_agents(agents_list)
    logger.debug('Arena created with objects added and new grid_matrix {}'.format(arena.grid_matrix))

    logger.info("Finished creating arena and agents. One item consumed due to overlap. Current item count is {}".format(
        arena.no_items))
    return arena, agents_list

def generate_from_file(filenameprefix,n_agents):
    """

    :param filename: file with full path
    :param n_agents: number of agents that form this.
    :return:
    """
    arenafilename = filenameprefix+'/arena.npy'
    grid_matrix = np.load(open(arenafilename, 'r'))
    logger.info("Initializing old configuration for grid and agents")
    logger.debug('creating arena object with grid_matrix {}'.format(grid_matrix))
    arena = config.ARENA_CURR(grid_matrix, visualize=config.VISUALIZE_SIM)
    # logger.debug('creating {} agents'.format(n_agents))
    agentsfilename = filenameprefix+'/agents.npy'
    agents_list = generate_agents_fromfile(agentsfilename,n_agents, arena)
    arena.init_add_agents(agents_list)
    logger.debug('Arena created with objects added and new grid_matrix {}'.format(arena.grid_matrix))

    logger.info("Finished creating arena and agents. One item consumed due to overlap. Current item count is {}".format(
        arena.no_items))
    return arena, agents_list

def generate_and_save(filenameprefix,size,n_agents,n_items):
    grid_matrix, final_numitems = generate_arena_matrix(size, n_items)
    if not os.path.exists(filenameprefix):
        os.makedirs(filenameprefix)
    np.save(open(filenameprefix+'/arena.npy','w+'),grid_matrix)
    logger.info("Initializing new configuration")
    assert np.shape(grid_matrix)[0] == size, "Grid_matrix not of size {}".format(size)
    agent_params = generate_agents_tosave(n_agents,grid_matrix)
    np.save(open(filenameprefix+'/agents.npy','w+'),agent_params)
    return 1

# generate_reload(3)
def generate_for_experiment(filenameprefix,n_experiments,n_agents,size,n_items):
    for i in range(n_experiments):
        fname = filenameprefix+'/'+str(i)
        generate_and_save(fname,size,n_agents,n_items)
    return

def generate_from_savedexperiment(filenameprefix,id,n_agents):
    fname = filenameprefix+'/'+str(id)
    return generate_from_file(fname,n_agents)

def gen_random_state(arena):
    dummyagent = generate_agents(1,arena,False)
    return dummyagent[0].__getstate__()
