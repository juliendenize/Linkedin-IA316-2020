import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dot, Flatten
from keras.models import Model


class RandomAgent():
    """ 
    Random agent.
    """

    def __init__(self, action_size, seed=None):
        """
        Initalize a random agent.

        Arguments
        ---------
        action_size: int
            the size of the action.
        seed: int
            the rng parameter for numpy.
        """
        self.action_size = action_size
        self._rng = np.random.RandomState(seed)
        
    def act(self, context):
        """
        Choose an action.

        Arguments
        ---------
        context: list()
            the context.
        
        Return
        ---------
        action: np.ndarray
            the action.
        """
        
        action = self._rng.choice([row[1] for row in context], self.action_size, replace=False) # note that action size is changing

        return action
        
    def update(self, context, action, reward):
        """
        Update the agent. Not relevent for this agent.
        """

        pass



class MLAgent():
    def __init__(self, action_size, ML_class, parameters = {"n_jobs" : -1}, online_learning = False, seed = None):
        """
        Initalize a Machine learning agent.

        Arguments
        ---------
        action_size: int
            the size of the action.
        ML_class: ModelClass
            the class of the machine learning model used for this agent.
        parameters: dict()
            the parameters for the model.
        online_learning: bool
            tells the agent to do or not do online learning
        seed: int
            the rng parameter for numpy.
        """
        self._rng = np.random.RandomState(seed = seed)

        self.action_size     = action_size
        self.ML_class        = ML_class
        self.parameters      = parameters
        self.online_learning = online_learning
        self.step_fit = 0

    def act(self, context):
        """
        Choose an action.

        Arguments
        ---------
        context: list()
            the context.
        
        Return
        ---------
        action: np.ndarray
            the action.
        """
        X = np.zeros((len(context), len(context[0][2])))

        for i, (_, _, variables) in enumerate(context):
            X[i, :] = variables

        predictions = self.model.predict(X)

        action  = np.argsort(predictions)[-self.action_size:][::-1].astype(int)

        #action = [context[idx][1] for idx in idx_action]

        return action


    def update(self, context, action, reward):
        """
        Update the agent. Not relevent for this agent because the online learning has not been implemented yet.
        """
        if self.online_learning:
            self.step_fit += 1

            X = np.zeros((self.action_size, self.X.shape[1]))
            y = np.zeros(self.action_size)
            for i in range(self.action_size):
                X[i] = context[action.astype(int)[i]][2]
                y[i] = reward[i]
            
            self.X = np.concatenate([self.X, X])
            self.y = np.concatenate([self.y, y])

            if self.step_fit % 100 == 0:
              self.model.fit(self.X, self.y)
        pass


    def train_agent(self, contexts, actions, rewards, clicks):
        """
        Train the agent using an history.

        Arguments
        ---------
        contexts: list()
            list of the history contexts.
        actions: np.ndarray
            the history of actions.
        rewards: np.ndarray
            the history of rewards.
        clicks: np.ndarray
            the history of clicks.
        """

        steps          = len(contexts)
        variables_size = len(contexts[0][0][2])

        self.X = np.zeros((steps * self.action_size, variables_size))
        self.y = np.zeros((steps * self.action_size))

        for step in range(steps):
            for i in range(self.action_size):
                self.X[step * self.action_size + i] = contexts[step][actions[step].astype(int)[i]][2]
                self.y[step * self.action_size + i] = rewards[step, i]

        self.model = self.ML_class(**self.parameters)
        self.model.fit(self.X, self.y)




class DeepAgent():
    def __init__(self, action_size, layers=[], parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 100},
                 online_learning = False, seed = None):
        """
        Initalize a Deep Learning agent.

        Arguments
        ---------
        action_size: int
            the size of the action.
        layers: list(int)
            the number of neurons for each layer.
        parameters: dict()
            the parameters for the fitting.
        online_learning: bool
            tells the agent to do or not do online learning
        seed: int
            the rng parameter for numpy.
        """
        self._rng = np.random.RandomState(seed = seed)

        self.action_size     = action_size
        self.layers          = layers
        self.parameters      = parameters
        self.online_learning = online_learning

        self.step_fit = 0
  
    def act(self, context):
        """
        Choose an action.

        Arguments
        ---------
        context: list()
            the context.

        Return
        ---------
        action: np.ndarray
            the action.
        """
        X = np.zeros((len(context), len(context[0][2])))

        for i, (_, _, variables) in enumerate(context):
            X[i, :] = variables

        predictions = self.model.predict(X)

        action  = np.argsort(predictions.flatten())[-self.action_size:][::-1].astype(int)

        #action = [context[idx][1] for idx in idx_action]

        return action


    def update(self, context, action, reward):
        """
        Update the agent. Not relevent for this agent because the online learning has not been implemented yet.
        """
        if self.online_learning:
            self.step_fit += 1

            X = np.zeros((self.action_size, self.X.shape[1]))
            y = np.zeros(self.action_size)
            for i in range(self.action_size):
                X[i] = context[action.astype(int)[i]][2]
                y[i] = reward[i]
            
            self.X = np.concatenate([self.X, X])
            self.y = np.concatenate([self.y, y])

            if self.step_fit % 100 == 0:
              print(self.step_fit)
              parameters = self.parameters.copy()
              parameters["epochs"] = 5

              self.model.fit(self.X, self.y, **parameters)
        pass


    def train_agent(self, contexts, actions, rewards, clicks):
        """
        Train the agent using an history.

        Arguments
        ---------
        contexts: list()
            list of the history contexts.
        actions: np.ndarray
            the history of actions.
        rewards: np.ndarray
            the history of rewards.
        clicks: np.ndarray
            the history of clicks.
        """

        steps          = len(contexts)
        variables_size = len(contexts[0][0][2])

        self.X = np.zeros((steps * self.action_size, variables_size))
        self.y = np.zeros((steps * self.action_size))

        for step in range(steps):
            for i in range(self.action_size):
                self.X[step * self.action_size + i] = contexts[step][actions[step].astype(int)[i]][2]
                self.y[step * self.action_size + i] = rewards[step, i]

        self.model = Sequential()
        self.model.add(Dense(self.layers[0], input_dim=variables_size))
        for layer in self.layers[1:]:
            self.model.add(Dense(layer))

        self.model.add(Dense(1, activation="linear"))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(self.X, self.y, **self.parameters)



class DotAgent():
    def __init__(self, action_size, seed = None):
        """
        Initalize a Machine learning agent.

        Arguments
        ---------
        action_size: int
            the size of the action.
        ML_class: ModelClass
            the class of the machine learning model used for this agent.
        parameters: dict()
            the parameters for the model.
        seed: int
            the rng parameter for numpy.
        """

        self.action_size = action_size

    def act(self, context):
        """
        Choose an action.

        Arguments
        ---------
        context: list()
            the context.
        
        Return
        ---------
        action: np.ndarray
            the action.
        """
        user_embeddings = np.zeros((len(context), (len(context[0][2]) - 2) // 2))
        item_embeddings = np.zeros((len(context), (len(context[0][2]) - 2) // 2))
        for i, (_, _, variables) in enumerate(context):
            user_embeddings[i, :] = variables[:(len(variables) - 2) // 2]
            item_embeddings[i, :] = variables[(len(variables) - 2) // 2: - 2]

        predictions = np.multiply(user_embeddings, item_embeddings).sum(axis = 1)

        action  = np.argsort(predictions)[-self.action_size:][::-1].astype(int)

        #action = [context[idx][1] for idx in idx_action]

        return action


    def update(self, context, action, reward):
        """
        Update the agent. Not relevent for this agent.
        """
        pass



class RegressionModel(Model):
    def __init__(self, embedding_size, max_user_id, max_item_id):
        super().__init__()
        
        self.user_embedding = Embedding(output_dim=embedding_size,
                                        input_dim=max_user_id + 1,
                                        input_length=1,
                                        name='user_embedding')
        self.item_embedding = Embedding(output_dim=embedding_size,
                                        input_dim=max_item_id + 1,
                                        input_length=1,
                                        name='item_embedding')
        
        # The following two layers don't have parameters.
        self.flatten = Flatten()
        self.dot = Dot(axes=1)
        
    def call(self, inputs):
        user_inputs = inputs[0]
        item_inputs = inputs[1]
        
        user_vecs = self.flatten(self.user_embedding(user_inputs))
        item_vecs = self.flatten(self.item_embedding(item_inputs))
        
        y = self.dot([user_vecs, item_vecs])
        return y    



class EmbeddingAgent():
    def __init__(self, action_size, embedding_size, max_user_id, max_item_id,
                 parameters = {"verbose": 1, "validation_split": 0.1, "callbacks": [EarlyStopping(patience = 10)], "epochs": 50, "batch_size": 64},
                 online_learning = False, seed = None):
      
        self.model = RegressionModel(embedding_size, max_user_id, max_item_id)
        self.model.compile(optimizer="adam", loss='mse')
        
        self.parameters      = parameters
        self.online_learning = online_learning
        self._rng            = np.random.RandomState(seed)
        self.action_size     = action_size

        self.step_fit = 0
        
    def train_agent(self, contexts, actions, rewards, clicks):
        steps        = len(contexts)

        self.users   = np.zeros((steps * self.action_size, 1))
        self.items   = np.zeros((steps * self.action_size, 1))
        self.y       = np.zeros((steps * self.action_size, 1))

        for step in range(steps):
            for i in range(self.action_size):
                self.users[step * self.action_size + i] = contexts[step][actions[step].astype(int)[i]][0]
                self.items[step * self.action_size + i] = contexts[step][actions[step].astype(int)[i]][1]
                self.y[step * self.action_size + i]     = rewards[step, i]

        self.model.fit([self.users, self.items], self.y, **self.parameters)
        
    def act(self, context):
        user        = np.asarray([row[0] for row in context])
        actions     = np.asarray([row[1] for row in context])
        predictions = self.model.predict([user, actions])
        action      = np.argsort(predictions.flatten())[-self.action_size:][::-1].astype(int)
        
        return action
    
    def update(self, context, action, reward):
        # online learning
        if self.online_learning:
            self.step_fit += 1

            users = np.zeros((self.action_size, 1))
            items = np.zeros((self.action_size, 1))
            y     = np.zeros((self.action_size, 1))
            for i in range(self.action_size):
                users[i] = context[action.astype(int)[i]][0]
                items[i] = context[action.astype(int)[i]][1]
                y[i]     = reward[i]
            
            self.users = np.concatenate([self.users, users])
            self.items = np.concatenate([self.items, items])
            self.y     = np.concatenate([self.y, y])

            if self.step_fit % 100 == 0:
              parameters = self.parameters.copy()
              parameters["epochs"] = 5

              self.model.fit([self.users, self.items], self.y, **parameters)

        pass