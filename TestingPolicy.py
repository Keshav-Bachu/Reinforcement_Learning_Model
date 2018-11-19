
import numpy as np
from Reinforcement_Learning_Model import ModelCalls as Model
from Reinforcement_Learning_Model import FormatData
from Reinforcement_Learning_Model import ReenforcementLearning as  REL
from Reinforcement_Learning_Model import ModelTrainHelper as helper


class PolicyGen:
    """Policy generator class for CtF env.
    
    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.
    
    Methods:
        gen_action: Required method to generate a list of actions.
        patrol: Private method to control a single unit.
    """
    
    def __init__(self, free_map, agent_list):
        """Constuctor for policy class.
        
        Patrolling policy provides the actions for the team of units that
        command units to approach the boarder between friendly and enemy
        zones and patrol along it.
        
        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.random = np.random
        #self.free_map = free_map 
        #self.heading_right = [True] * len(agent_list) #: Attr to track directions.
        
    def gen_action(self, agent_list, observation, free_map=None):
        action_out = []
        #get the weights, biases, and QW
        weights = np.load("Reinforcement_Learning_Model/Weights/Itteration 1/weights.npy")
        biases = np.load("Reinforcement_Learning_Model/Weights/Itteration 1/biases.npy")
        QW = np.load("Reinforcement_Learning_Model/Weights/Itteration 1/QW.npy")
        
        for idx,agent in enumerate(agent_list):
            a = self.generateAction(agent, observation, weights, biases, QW)
            action_out.append(a)

        return action_out
        

    def generateAction(self, agent, obs, weights, biases, QW):
        """Generate 1 action for given agent object."""
        x,y = agent.get_loc()
        locArray = np.asanyarray([y,x])
        action = Model.generateFromLocation(obs, locArray, weights, biases, QW)
        
        #print(action[0])
        return action[0]
