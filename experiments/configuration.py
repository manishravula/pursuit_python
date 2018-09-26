import time
import inspect


DEBUG=False

NO_MOVEMENTS = 4
LOAD_ACTION_PROBABILITY = []

DIMENSIONS = 5

SIM_DELAY = 0.0

#Agents to use

#Arena to use
#src.generation_init config
COOPERATION_INDEX = .8 # index*grid_min
NO_TYPES = 4
N_AGENTS = 3
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
VISUALIZE_SIM = False


#MCTS Related
N_ROLLOUTS = 5
MAX_ROLLOUT_DEPTH = 3
MAX_HEURISTIC_ROLLOUT_DEPTH = 3



#Twilio details
from experiments import twiliocreds
SMSClient = twiliocreds.SMSClient
from_number = "+15126438645"
to_number = "+15125022558"

#Experimental details
N_EXPERIMENTS = 10
N_MAXITERS_IN_EXPERIMENTS = 100


#Champ BACKTRACKING
N_CHAMP_SKIPSTEPS = 10


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
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': './logging/debug_log_{}-{}.txt'.format(time.asctime().replace(' ',''),inspect.stack()[-1][1].replace('/',''))
        },
    },
    'loggers': {
        '': {
            'handlers': ['infoterminal','Debug_File'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}
