"""
Project:    TII - MAS Fault Detection, Identification, and Reconfiguration
Author:     Yifan Guo
Description:
            - Implementation of " Distributed l1-State-and-Fault Estimation for Multiagent Systems".
"""
from dynamic_models.generic_agent import GenericAgent as Agent
import numpy as np
import cvxpy as cp


def distributed_l1(agents, iters_max, edges, u, y, previous_est):
    zeta = 0.1
    num_agents = len(agents)
    M1, M2 = divide_group(edges)
    # Initilization
    agents[0].init_est_all(num_agents)
    previous_est = []
    for id in range(num_agents):
        agents[id].init_Mu(num_agents)

    # ADMM
    for iter in range(iters_max):
        for id in M1:
            u = np.ones((12,1)) # Should change to real inputs
            neighbors = [agents[i] for i in agents[id].get_neighbors()]
            agents[id].v = agents[id].Mu + zeta * np.sum(map(lambda agent:agent.est_all, neighbors), axis=0)
            est = cp.Variable(agents[id].est_all.shape)
            obj = (cp.norm(est, 1) - (A@agents[id].est_all+cp.multiply(B, u)))/num_agents + agents[id].v.T@est + agents[id].v*(zeta/2)*cp.norm(est,2)
            if id == 0:
                constraints = [y[id]==C@est]
            else:
                constraints = [y[id]==C1@est]
            prob = cp.Problem(obj, constraints)
            prob.solve()
            agents[id].est_all = est.value
            broadcast(agents, id, agents[id].est_all)
        
        for id in M2:
            u = np.ones((12,1)) # Should change to real inputs
            neighbors = [agents[i] for i in agents[id].get_neighbors()]
            agents[id].v = agents[id].Mu + zeta * np.sum(map(lambda agent:agent.est_all, neighbors), axis=0)
            est = cp.Variable(agents[id].est_all.shape)
            obj = (cp.norm(est, 1) - (A@agents[id].est_all+cp.multiply(B, u)))/num_agents + agents[id].v.T@est + agents[id].v*(zeta/2)*cp.norm(est,2)
            if id == 1:
                constraints = [y[id]==C@est]
            else:
                constraints = [y[id]==C1@est]
            prob = cp.Problem(obj, constraints)
            prob.solve()
            agents[id].est_all = est.value
            broadcast(agents, id, agents[id].est_all)
        
    

        for id in range(num_agents):
            agents[id].Mu = agents[id].Mu + zeta*agents[id].get_discrepancy(agents)
    
    for id in range(num_agents):
        agents[id].error = agents[id].est_all - previous_est[id]

    return [agents[id].error for id in range(num_agents)]




     

def divide_group(edges):
    '''
    Divide agents into two groups for ADMM, each directed edge starts from an agent in M1(M2) to an agent in M2(M1)
    '''
    # Later change to a real spliting algorithm, now just hard code
    M1 = np.array([0, 2, 4])
    M2 = np.array([1, 3, 5])
    return M1, M2

def broadcast(agents, id, est_all):
    neighbors = agents[id].get_neighbors()
    for i in neighbors:
        agents[i].est_all = est_all



if __name__=='__main__':
    num_agents      =   6
    num_faulty      =   1   # must be << num_agents for sparse error assumption
    agents      =   [None] * num_agents
    d           =   6   # hexagon side length
    agents[0]   =   Agent(agent_id= 0,
                        init_position= np.array([[d/2, -d*np.sqrt(3)/2]]).T)
    agents[1]   =   Agent(agent_id= 1,
                        init_position= np.array([[-d/2, -d*np.sqrt(3)/2]]).T)
    agents[2]   =   Agent(agent_id= 2,
                        init_position= np.array([[-d, 0]]).T)
    agents[3]   =   Agent(agent_id= 3,
                        init_position= np.array([[-d/2, d*np.sqrt(3)/2]]).T)
    agents[4]   =   Agent(agent_id= 4,
                        init_position= np.array([[d/2, d*np.sqrt(3)/2]]).T)
    agents[5]   =   Agent(agent_id= 5,
                        init_position= np.array([[d, 0]]).T)
    # Set Neighbors
    edges       =   [[0,1], [0,2], [0,3], 
                    [0,4], [0,5], [1,2],
                    [1,3], [1,4], [1,5],
                    [2,3], [2,4], [2,5],
                    [3,4], [3,5], [4,5],
                    
                    [1,0], [2,0], [3,0], 
                    [4,0], [5,0], [2,1],
                    [3,1], [4,1], [5,1],
                    [3,2], [4,2], [5,2],
                    [4,3], [5,3], [5,4]] # these edges are directed
    neighbors = [[] for i in range(num_agents)]

    # Simple test dynamics, each drone has 2D position as states (totally 2*6=12 dimension state space), the speed sololy controlled by control inputs,
    # system outputs are the states.
    A = np.zeros((12,12))
    B = np.ones((12,1))
    C = np.eye(12)
    C1 = np.eye(12)
    for v1, v2 in edges:
        if v2 not in neighbors[v1]:
            neighbors[v1].append(v2)
    
    for id in range(num_agents):
        agents[id].init_est_all(num_agents)
        agents[id].set_neighbors(neighbors[id])
    