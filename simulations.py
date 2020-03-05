import numpy as np
import pickle
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from Environment import Environment
from Agents import RandomAgent, MLAgent, DeepAgent, EmbeddingAgent, DotAgent
from tensorflow.keras.callbacks import EarlyStopping
from School import School


def run_exp(agent, env, nb_steps, action_size, env_seed):
    """
    Run an experiment for an agent.

    Arguments
    ---------
    agent: Agent
        the agent.
    env: Environment
        the environment.
    nb_steps: int
        number of steps for the experiment.
    action_size: int
        the action size.
    env_seed: int
        seed for the rng of the environment.

    Return
    -------
    dict
        the results of the experiment.
    """
    rewards = np.zeros((nb_steps, action_size))
    regrets = np.zeros((nb_steps, action_size))
    actions = np.zeros((nb_steps, action_size))
    clicks  = np.zeros((nb_steps, action_size))
    context = env.reset(env_seed)

    for i in range(nb_steps):
        # Select action from agent policy.
        action = agent.act(context)
        
        # Play action in the environment and get reward.
        next_context, reward, click, best_reward  = env.step(action)
        
        # Update agent.
        agent.update(context, action, reward)
        context = next_context
        
        # Save history.
        rewards[i] = reward
        actions[i] = action
        regrets[i] = best_reward - reward

    reward = rewards.sum(axis = 0)
    regret = np.sum(regrets, axis = 0)
    return {'reward': reward, 
            'regret': regret,
            'rewards': rewards,
            'regrets': regrets,
            'actions': actions,
            'cum_rewards': np.cumsum(rewards, axis = 0), 
            'cum_regrets': np.cumsum(regrets, axis = 0),
            }



def run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, env_seed):
    """
    Run an experiment for an agent with an history made by a random agent.

    Arguments
    ---------
    random_agent: RandomAgent
        the random agent.
    agent: Agent
        the agent.
    env: Environment
        the environment.
    nb_steps_history: int
        number of steps for the history.
    nb_steps: int
        number of steps for the experiment.
    action_size: int
        the action size.
    env_seed: int
        seed for the rng of the environment.

    Return
    -------
    dict
        the results of the experiment.
    """
    history = np.zeros((nb_steps_history, 4))
    rewards = np.zeros((nb_steps, action_size))
    regrets = np.zeros((nb_steps, action_size))
    actions = np.zeros((nb_steps, action_size))
    clicks  = np.zeros((nb_steps, action_size))

    contexts_hist = []
    rewards_hist = np.zeros((nb_steps_history, action_size))
    actions_hist = np.zeros((nb_steps_history, action_size))
    clicks_hist  = np.zeros((nb_steps_history, action_size))

    context = env.reset(env_seed)

    for i in range(nb_steps_history):
        # Select action from agent policy.
        action = random_agent.act(context)
        
        # Play action in the environment and get reward.
        next_context, reward, click, best_reward = env.step(action)
        
        # Update history
        rewards_hist[i] = reward
        actions_hist[i] = action
        clicks_hist[i]  = click
        contexts_hist   += [context]
        
        # Update agent.
        random_agent.update(context, action, reward)
        context = next_context
        
    agent.train_agent(contexts_hist, actions_hist, rewards_hist, clicks_hist)
    
    for i in range(nb_steps):
        # Select action from agent policy.
        action = agent.act(context)
        
        # Play action in the environment and get reward.
        next_context, reward, click, best_reward  = env.step(action)

        
        # Update agent.
        agent.update(context, action, reward)
        context = next_context
        
        # Save history.
        #context[i] = context
        rewards[i] = reward
        actions[i] = action
        regrets[i] = best_reward - reward


    reward = rewards[:, 0].sum()
    regret = np.sum(regrets, axis = 0)

    return {'reward': reward, 
            'regret': regret,
            'rewards': rewards,
            'regrets': regrets,
            'actions': actions,
            'cum_rewards': np.cumsum(rewards, axis = 0), 
            'cum_regrets': np.cumsum(regrets, axis = 0),
            }



def run_several_experiments_Random(evolutive_env = False, nb_exp = 10, nb_steps = 1000, action_size = 5):
    """
    Run several experiments for the Random agent.
    
    Arguments
    ---------
    nb_exp: int
        number of experiences.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """
    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env   = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive=evolutive_env)
        agent = RandomAgent(action_size, seed=i)
        exp   = run_exp(agent, env, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal



def run_several_experiments_hist_ML(ML_class, evolutive_env = False, nb_exp = 10, nb_steps_history = 500, nb_steps = 1000, action_size = 5, agent_parameters = {"n_jobs" : -1}):
    """
    Run several experiments for the ML agent.
    
    Arguments
    ---------
    ML_class: Class
        the class for the machine learning algorithm.
    evolutive_env: bool
        indicates if the environment is evolutive.
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of hist steps.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    agent_parameters: dict()
        the agent parameters.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """

    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env   = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = MLAgent(action_size, ML_class, parameters = agent_parameters, seed = i)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal



def run_several_experiments_hist_DL(layers, nb_exp = 10, nb_steps_history = 500, nb_steps = 1000, action_size = 5, evolutive_env = False,
                                    parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 100, "batch_size": 64}):
    """
    Run several experiments for the DL agent.
    
    Arguments
    ---------
    layers: list()
        number of neurons for each layer
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of hist steps.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    evolutive_env: bool
        indicates if the environment is evolutive.
    parameters: dict()
        the agent parameters.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """

    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env          = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = DeepAgent(action_size, layers, parameters, seed = i)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal


def run_several_experiments_hist_ML_online(ML_class, evolutive_env = False, nb_exp = 10, nb_steps_history = 500, nb_steps = 1000, action_size = 5, agent_parameters = {"n_jobs" : -1}):
    """
    Run several experiments for the online learning ML agent.
    
    Arguments
    ---------
    ML_class: Class
        the class for the machine learning algorithm.
    evolutive_env: bool
        indicates if the environment is evolutive.
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of hist steps.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    parameters: dict()
        the agent parameters.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """
    
    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env   = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive=evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = MLAgent(action_size, ML_class, parameters = agent_parameters, seed = i, online_learning = True)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal
    

def run_several_experiments_hist_DL_online(layers, nb_exp = 10, nb_steps_history = 500, nb_steps = 1000, action_size = 5, evolutive_env = False,
                                           parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 100, "batch_size": 64}):
    """
    Run several experiments for the DL agent with online learning.
    
    Arguments
    ---------
    layers: list()
        number of neurons for each layer
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of hist steps.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    evolutive_env: bool
        indicates if the environment is evolutive.
    parameters: dict()
        the agent parameters.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """
    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env          = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = DeepAgent(action_size, layers, parameters, seed = i, online_learning = True)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal


def run_several_experiments_Dot(evolutive_env = False, nb_exp = 10, nb_steps = 1000, action_size = 5):
    """
    Run several experiments for the Dot agent.
    
    Arguments
    ---------
    evolutive_env: bool
        indicates if the environment is evolutive.
    nb_exp: int
        number of experiences.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """

    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")

        env   = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        agent = DotAgent(action_size, seed=i)
        exp   = run_exp(agent, env, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal


def run_several_experiments_hist_Embedding(evolutive_env = False, nb_exp = 10, nb_steps_history = 1000, nb_steps = 1000, action_size = 5,
                                           parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 50, "batch_size": 64}):
    """
    Run several experiments for the embedding agent.
    
    Arguments
    ---------
    evolutive_env: bool
        indicates if the environment is evolutive.
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of steps for history.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    parameters: dict()
        the parameters for the model.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """

    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env          = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = EmbeddingAgent(action_size, 6, env.n_users_0 - 1, env.n_offers_0, parameters = parameters, seed=i)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal


def run_several_experiments_hist_Embedding_online(evolutive_env = False, nb_exp = 10, nb_steps_history = 1000, nb_steps = 1000, action_size = 5,
                                           parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 50, "batch_size": 64}):
    """
    Run several experiments for the embedding agent with online learning.
    
    Arguments
    ---------
    evolutive_env: bool
        indicates if the environment is evolutive.
    nb_exp: int
        number of experiences.
    nb_steps_history: int
        number of steps for history.
    nb_steps: int
        number of steps.
    action_size: int
        the action size.
    parameters: dict()
        the parameters for the model.
    
    Return
    --------
    regret: np.ndarray
        the regret
    regrets: np.ndarray
        the cumulative regrets.
    rewards: np.ndarray
        the rewards
    regrets_normal: np.ndarray
        the normal regrets.
    """
    regret = np.zeros((nb_exp, action_size))
    regrets = np.zeros((nb_exp, nb_steps, action_size))
    rewards = np.zeros((nb_exp, nb_steps, action_size))
    regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

    time1 = time.time()
    for i in range(nb_exp):
        if i %  10 ** (np.log10(nb_exp) - 1) == 0:
            print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")
    
        env          = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels, skills, places, evolutive = evolutive_env)
        random_agent = RandomAgent(action_size, seed=i)

        agent = EmbeddingAgent(action_size, 6, env.n_users_0 - 1, env.n_offers_0, parameters = parameters, seed=i, online_learning=True)
        exp   = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)

        regret[i]  = exp['regret'] 
        regrets[i] = exp['cum_regrets']
        regrets_normal[i] = exp['regrets']
        rewards[i] = exp['rewards']

    print(f"End of the simulations, time elapsed: {time.time() - time1:.3f} seconds.")
    return regret, regrets, rewards, regrets_normal


def run_several_experiments_several_agents(agent_dict, action_size, evolutive_env, folder_saves, schools_dictionnary, domains_to_skills_dictionnary,
                                           companies, hierarchical_levels, skills, places, nb_exp = 20, nb_steps = 1000,
                                           nb_steps_history = 1000):
  
    env = Environment(schools_dictionnary, domains_to_skills_dictionnary, companies, hierarchical_levels,
                      skills, places, n_users = 30, n_offers = 30, evolutive = evolutive_env)

    for agent_name in agent_dict:
        print(f"{agent_name} started.")

        if os.path.exists(os.path.join(folder_saves, f"_rewards_{agent_name}.pkl")):
            continue

        regret = np.zeros((nb_exp, action_size))
        regrets = np.zeros((nb_exp, nb_steps, action_size))
        rewards = np.zeros((nb_exp, nb_steps, action_size))
        regrets_normal = np.zeros((nb_exp, nb_steps, action_size))

        time1 = time.time()
        for i in range(nb_exp):
            if i %  5 == 0:
                print(f"Simulation {i} started, time elapsed: {(time.time() - time1):.3f} seconds.")

            if "random" in agent_name:
                agent = RandomAgent(**agent_dict[agent_name], seed = i)

            elif "ML" in agent_name:
                agent = MLAgent(**agent_dict[agent_name], seed = i)

            elif "DL" in agent_name:
                agent = DeepAgent(**agent_dict[agent_name], seed = i)

            elif "Dot" in agent_name:
                agent = DotAgent(**agent_dict[agent_name], seed = i)

            elif "Embedding" in agent_name:
                agent = EmbeddingAgent(**agent_dict[agent_name], seed = i)  


            if "random" in agent_name or "Dot" in agent_name:
                exp = run_exp(agent, env, nb_steps, action_size, i)

            else:
                random_agent = RandomAgent(agent.action_size, seed = i) 
                exp          = run_exp_hist(random_agent, agent, env, nb_steps_history, nb_steps, action_size, i)
            

            regret[i]         = exp['regret'] 
            regrets[i]        = exp['cum_regrets']
            regrets_normal[i] = exp['regrets']
            rewards[i]        = exp['rewards']
        
        with open(os.path.join(folder_saves, f"_regret_{agent_name}.pkl"),"wb") as f:
            pickle.dump(regret,f)
        
        with open(os.path.join(folder_saves, f"_regrets_{agent_name}.pkl"),"wb") as f:
            pickle.dump(regrets,f)
        
        with open(os.path.join(folder_saves, f"_regrets_normal_{agent_name}.pkl"),"wb") as f:
            pickle.dump(regrets_normal,f)
        
        with open(os.path.join(folder_saves, f"_rewards_{agent_name}.pkl"),"wb") as f:
            pickle.dump(rewards,f)   



def plot_regret(regret, regrets):
    plt.plot(regrets[:, :, 0].mean(axis=0), color='blue')
    plt.plot(np.quantile(regrets[:, :, 0], 0.05, axis=0), color='grey', alpha=0.5)
    plt.plot(np.quantile(regrets[:, :, 0], 0.95, axis=0), color='grey', alpha=0.5)
    plt.title('Mean regret: {:.2f}'.format(regret[:, 0].mean(axis = 0)))
    plt.xlabel('steps')
    plt.ylabel('regret')
    plt.grid()
    plt.show()


domains_to_skills_dictionnary = {
    "Computer Science": ["Symfony", "Flask", "Docker", "SQL", "Python"],
    "Data Science":     ["Machine Learning", "Deep Learning", "Pandas", "Scikit-Learn", "Office"]
}

schools_dictionnary = {
    "Télécom Paris"   : {"domains": ["Data Science"],
                         "place"  : "Paris"},
    "ENSIEE"          : {"domains": ["Computer Science"],
                         "place"  : "Evry"},
}

schools_dictionnary = {school: School(school, schools_dictionnary[school]["place"], schools_dictionnary[school]["domains"]) for school in schools_dictionnary}

companies = ["Société Générale", "BNP Paribas", "Wavestone", "CEA", "INRIA"]

hierarchical_levels = ["intern", "junior", "senior"]

skills = [skill for domain in domains_to_skills_dictionnary for skill in domains_to_skills_dictionnary[domain]]

places = ["Evry", "Paris", "Saclay"]

companies_oh = OneHotEncoder(sparse=False).fit(np.asarray(companies).reshape(-1, 1))
skills_oh    = OneHotEncoder(sparse=False).fit(np.asarray(skills).reshape(-1, 1))
places_oh    = OneHotEncoder(sparse=False).fit(np.asarray(places).reshape(-1, 1))
domains_oh   = OneHotEncoder(sparse=False).fit(np.asarray(list(domains_to_skills_dictionnary.keys())).reshape(-1, 1))