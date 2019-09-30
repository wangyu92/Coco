import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class A3CModel:
    def __init__(self, state_dim, action_dim, model=None):
        """
        [params]
        model : to derive the parameters of global networks
        """
        
        # Hyper params
        self.discount_factor = 0.99
        self.learning_rate_actor = 2.5e-4
        self.learning_rate_critic = 2.5e-4
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor, self.critic = self.build_model(model)
        
    def build_model(self, model=None):
        inputs = keras.layers.Input(shape=self.state_dim)
        fc1 = keras.layers.Dense(10, activation='relu')(inputs)
        fc2 = keras.layers.Dense(10, activation='relu')(fc1)
        
        policy = keras.layers.Dense(self.action_dim[0], activation='softmax')(fc2)
        value = keras.layers.Dense(1, activation='linear')(fc2)
        
        actor = keras.Model(inputs=inputs, outputs=policy)
        critic = keras.Model(inputs=inputs, outputs=value)
        
        actor._make_predict_function()
        critic._make_predict_function()
        
        if model != None:
            actor.set_weights(model.actor.get_weights())
            critic.set_weights(model.critic.get_weights())

        # 테스트로 생성한 레이들을 출력해줌
        if model == None:
            actor.summary()
            critic.summary()
        
        return actor, critic
    
    def actor_optimizer(self):
        action = keras.backend.placeholder(shape=[None, self.action_dim[0]])
        advantages = keras.backend.placeholder(shape=[None, ])
        
        policy = self.actor.output
        
        # policy cross entropy loss function
        action_prob = keras.backend.sum(action * policy, axis=1)
        cross_entropy = keras.backend.log(action_prob + 1e-10) * advantages
        cross_entropy = -keras.backend.sum(cross_entropy)
        
        # cross entropy error for exploration
        entropy = keras.backend.sum(policy * keras.backend.log(policy + 1e-10), axis=1)
        entropy = keras.backend.sum(entropy)
        
        # final loss function
        loss = cross_entropy + 0.01 * entropy
        
        optimizer = keras.optimizers.RMSprop(learning_rate=self.learning_rate_actor, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(params=self.actor.trainable_weights, loss=loss)
        train = keras.backend.function([self.actor.input, action, advantages], [loss], updates=updates)
        
        return train
    
    def critic_optimizer(self):
        discounted_prediction = keras.backend.placeholder(shape=[None, ])
        
        value = self.critic.output
        
        # loss function by square of [G - value]
        loss = keras.backend.mean(keras.backend.square(discounted_prediction - value))
        
        optimizer = keras.optimizers.RMSprop(lr=self.learning_rate_critic, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(params=self.critic.trainable_weights, loss=loss)
        train = keras.backend.function([self.critic.input, discounted_prediction], [loss], updates=updates)
        
        return train
    
    def get_action(self, state):
        policy = self.actor.predict(np.expand_dims(state, axis=0))
        action_index = np.random.choice(self.action_dim[0], 1, p=policy)[0]
        return action_index, policy
    
    
    
    
    def lead_model(self, path):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")
        
    def save_model(self, path):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")