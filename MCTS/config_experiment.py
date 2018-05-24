import time
import inspect

NO_MOVEMENTS = 4
LOAD_ACTION_PROBABILITY = []

NO_TYPES = 4
N_AGENTS = 3



#Initialization stuff
COOPERATION_INDEX = .8 # index*grid_min


#Agents to use
# from src.agent_expressive1 import Agent as AGENT_CURR

#Arena to use

#Initialization to use
FROM_MEMORY = 1
FROM_NEW = 0
INIT_TYPE = FROM_NEW
# INIT_TYPE = 'random'

#Visualization stuff

#visualization of the estimation process
VISUALIZE_ESTIMATION = False

#Save figures or not switch
VISUALIZE_ESTIMATION_SAVE = False
DPI = 300

#Visualize the simulation or not.
VISUALIZE_SIM = True

#logging config
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'infoterminal': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            # 'stream': 'sys.stdout'
        },
        'Debug_File':{
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': '../logging/debug_log_{}-{}.txt'.format(time.asctime().replace(' ',''),inspect.stack()[-1][1].replace('/',''))
        },
    },
    'loggers': {
        '': {
            'handlers': ['infoterminal','Debug_File'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
