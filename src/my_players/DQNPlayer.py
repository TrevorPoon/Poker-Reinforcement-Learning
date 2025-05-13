from pypokerengine.players import BasePokerPlayer
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

# hyper-parameters
batch_size = 64
learning_rate = 1e-4
gamma = 0.99
exp_replay_size = 100000
learn_start = 500
target_net_update_freq = 10000
epsilon = 0.2
decay_factor = 0.9
final_epsilon = 0.001

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return rand.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Deep Q Network
class DQN(nn.Module):

    def __init__(self, input_shape, num_actions, lstm_hidden_size=128):
        super(DQN, self).__init__()

        self.input_dim = input_shape[0]
        self.num_actions = num_actions
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(self.lstm_hidden_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, self.num_actions)

    def forward(self, x, hidden=None):
        if x.ndim == 2:  # If (batch_size, input_dim)
            x = x.unsqueeze(1)  # Reshape to (batch_size, 1, input_dim)
        
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out_last_step = lstm_out[:, -1, :]

        x = F.relu(self.fc1(lstm_out_last_step))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x, hidden


class DQNPlayer(BasePokerPlayer):

    def __init__(self, model_path, optimizer_path, training):
        """
        State: hole_card, community_card, self.stack, opponent_player.action
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # training device: cpu > cuda
        self.nb_player = None # Number of players in the game 
        self.player_id = None # DQN Player ID (ID = 0 if p1) 
        self.episode = 0 # Sequence of action
        self.declare_memory()
        self.loss = None # Model loss

        # hyper-parameter for Deep Q Learning
        self.epsilon = self.max_epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.experience_replay_size = exp_replay_size
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.target_net_update_freq = target_net_update_freq

        # training-required game attribute
        self.stack = 100
        self.hole_card = None
        self.model_path = model_path
        self.optimizer_path = optimizer_path
        self.update_count = 0
        self.history = []
        self.training = training
        self.lstm_hidden_size = 128 # Define LSTM hidden size
        self.policy_hidden = None # Hidden state for policy network
        self.target_hidden = None # Hidden state for target network

        # Player's stats
        self.hand_count = 0
        self.VPIP = 0
        self.last_vpip_action = None
        self.PFR = 0
        self.last_pfr_action = None
        self.three_bet = 0
        self.last_3_bet_action = None
        self.street_first_action = None
        self.raisesizes = [0.25, 0.33, 0.5, 0.75, 1, 1.25, 1.5]
        self.last_reward = 0
        self.accumulated_reward = 0
        self.state = []
        self.action_stat = {
            'preflop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'flop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'turn': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'river': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0}
        }

        self.hand_type = None
        self.card_reward_stat = {
                # Pocket Pairs
                'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                
                # Suited Hands (AKs down to 32s)
                'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                '54s': 0, '53s': 0, '52s': 0,
                '43s': 0, '42s': 0,
                '32s': 0,
                
                # Offsuit Hands (AKo down to 32o)
                'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                '54o': 0, '53o': 0, '52o': 0,
                '43o': 0, '42o': 0,
                '32o': 0
            }

        self.card_action_stat = {
            'preflop': {
                'raise': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                },
                'call': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'check': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'fold': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }
            },
            'flop': {
                'raise': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                },
                'call': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'check': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'fold': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }
            },
            'river': {
                'raise': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                },
                'call': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'check': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'fold': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }
            },
            'turn': {
                'raise': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                },
                'call': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'check': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }, 
                'fold': {
                    # Pocket Pairs
                    'AA': 0, 'KK': 0, 'QQ': 0, 'JJ': 0, 'TT': 0, '99': 0, '88': 0, '77': 0, '66': 0, '55': 0, '44': 0, '33': 0, '22': 0,
                    
                    # Suited Hands (AKs down to 32s)
                    'AKs': 0, 'AQs': 0, 'AJs': 0, 'ATs': 0, 'A9s': 0, 'A8s': 0, 'A7s': 0, 'A6s': 0, 'A5s': 0, 'A4s': 0, 'A3s': 0, 'A2s': 0,
                    'KQs': 0, 'KJs': 0, 'KTs': 0, 'K9s': 0, 'K8s': 0, 'K7s': 0, 'K6s': 0, 'K5s': 0, 'K4s': 0, 'K3s': 0, 'K2s': 0,
                    'QJs': 0, 'QTs': 0, 'Q9s': 0, 'Q8s': 0, 'Q7s': 0, 'Q6s': 0, 'Q5s': 0, 'Q4s': 0, 'Q3s': 0, 'Q2s': 0,
                    'JTs': 0, 'J9s': 0, 'J8s': 0, 'J7s': 0, 'J6s': 0, 'J5s': 0, 'J4s': 0, 'J3s': 0, 'J2s': 0,
                    'T9s': 0, 'T8s': 0, 'T7s': 0, 'T6s': 0, 'T5s': 0, 'T4s': 0, 'T3s': 0, 'T2s': 0,
                    '98s': 0, '97s': 0, '96s': 0, '95s': 0, '94s': 0, '93s': 0, '92s': 0,
                    '87s': 0, '86s': 0, '85s': 0, '84s': 0, '83s': 0, '82s': 0,
                    '76s': 0, '75s': 0, '74s': 0, '73s': 0, '72s': 0,
                    '65s': 0, '64s': 0, '63s': 0, '62s': 0,
                    '54s': 0, '53s': 0, '52s': 0,
                    '43s': 0, '42s': 0,
                    '32s': 0,
                    
                    # Offsuit Hands (AKo down to 32o)
                    'AKo': 0, 'AQo': 0, 'AJo': 0, 'ATo': 0, 'A9o': 0, 'A8o': 0, 'A7o': 0, 'A6o': 0, 'A5o': 0, 'A4o': 0, 'A3o': 0, 'A2o': 0,
                    'KQo': 0, 'KJo': 0, 'KTo': 0, 'K9o': 0, 'K8o': 0, 'K7o': 0, 'K6o': 0, 'K5o': 0, 'K4o': 0, 'K3o': 0, 'K2o': 0,
                    'QJo': 0, 'QTo': 0, 'Q9o': 0, 'Q8o': 0, 'Q7o': 0, 'Q6o': 0, 'Q5o': 0, 'Q4o': 0, 'Q3o': 0, 'Q2o': 0,
                    'JTo': 0, 'J9o': 0, 'J8o': 0, 'J7o': 0, 'J6o': 0, 'J5o': 0, 'J4o': 0, 'J3o': 0, 'J2o': 0,
                    'T9o': 0, 'T8o': 0, 'T7o': 0, 'T6o': 0, 'T5o': 0, 'T4o': 0, 'T3o': 0, 'T2o': 0,
                    '98o': 0, '97o': 0, '96o': 0, '95o': 0, '94o': 0, '93o': 0, '92o': 0,
                    '87o': 0, '86o': 0, '85o': 0, '84o': 0, '83o': 0, '82o': 0,
                    '76o': 0, '75o': 0, '74o': 0, '73o': 0, '72o': 0,
                    '65o': 0, '64o': 0, '63o': 0, '62o': 0,
                    '54o': 0, '53o': 0, '52o': 0,
                    '43o': 0, '42o': 0,
                    '32o': 0
                }
            }
        }

        # declare DQN model
        self.num_actions = 11 # 2 + 2 + len(self.raisesizes) # Fold, Call, Min Raise, Max Raise (Allin) and pot size raises
        self.num_feats = (79, ) # Updated num_feats: 46 (base) + 1 (active_players) + 1 (eff_stack) + 1 (pot_odds) + 30 (street_action_hist)
        self.declare_networks()
        try:
            self.policy_net.load_state_dict(torch.load(self.model_path, weights_only=True))
        except:
            pass
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net = self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        try:
            self.optimizer.load_state_dict(torch.load(self.optimizer_path, weights_only=True))
        except:
            pass
        self.losses = []
        self.sigma_parameter_mag = []
        if self.training:
            self.policy_net.train()
            self.target_net.train()
        else:
            self.policy_net.eval()
            self.target_net.eval()
        

    def declare_networks(self):
        self.policy_net = DQN(self.num_feats, self.num_actions, lstm_hidden_size=self.lstm_hidden_size)
        self.target_net = DQN(self.num_feats, self.num_actions, lstm_hidden_size=self.lstm_hidden_size)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(exp_replay_size)
        return self.memory

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def save_sigma_param_magnitudes(self):
        tmp = []
        for name, param in self.policy_net.named_parameters():
            if param.requires_grad:
                if 'sigma' in name:
                    tmp += param.data.cpu().numpy().ravel().tolist()
        if tmp:
            self.sigma_parameter_mag.append(np.mean(np.abs(np.array(tmp))))

    def save_loss(self, loss):
        self.losses.append(loss)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        # estimate
        current_q_values_full, _ = self.policy_net(batch_state) # Hidden state initialized to None by default in forward
        current_q_values = current_q_values_full.gather(1, batch_action)

        # target
        with torch.no_grad():
            # To prevent tracking history of gradient
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                # Use target_net for next state Q values
                target_q_full, _ = self.target_net(non_final_next_states) # Hidden state initialized to None
                max_next_action = self.get_max_next_state_action(non_final_next_states, target_q_full) # Pass Q values if needed by selection
                max_next_q_values[non_final_mask] = target_q_full.gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(current_q_values, expected_q_values)
        self.loss = loss.detach().item()
        # print(loss)
        # diff = (expected_q_values - current_q_values)
        # loss = self.huber(diff)
        # loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, episode=0):
        if not self.training:
            return None

        self.append_to_replay(s, a, r, s_)

        if episode < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        # self.save_loss(loss.item())
        # self.save_sigma_param_magnitudes()

    def update_target_model(self):
        """
        to use in fix-target
        """
        if self.update_count % self.target_net_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print("update target: ", self.update_count)

            if self.accumulated_reward - self.last_reward > 0:
                self.epsilon = max(final_epsilon, self.epsilon * decay_factor)
            else:
                self.epsilon = min(self.max_epsilon, self.epsilon / decay_factor)
                                   
            self.last_reward = self.accumulated_reward

    def get_max_next_state_action(self, next_states, target_q_values=None):
        # If target_q_values are already computed, use them directly
        if target_q_values is not None:
            return target_q_values.max(dim=1)[1].view(-1, 1)
        # Otherwise, compute them (this path might be redundant if compute_loss always passes them)
        q_values, _ = self.target_net(next_states) # Hidden state will be new
        return q_values.max(dim=1)[1].view(-1, 1)

    @staticmethod
    def huber(x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    @staticmethod
    def bce_loss(x, y):
        """
        :return: binary entropy loss between x and y
        """
        _x = 1 / (1 + torch.exp(-x))
        _y = 1 / (1 + torch.exp(-y))
        return -(_y * torch.log(_x) + (1 - _y) * torch.log(1 - _x))

    @staticmethod
    def card_to_int(card):
        """convert card to int, card[0]:suit, card[1]:rank"""
        suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                    'A': 12}
        return suit_map[card[0]] * 13 + rank_map[card[1]]
    
    def get_hand_type(self, hole_card):

        # Sort cards by rank (for consistency)
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        card1, card2 = sorted(hole_card, key=lambda card: ranks[card[1]], reverse=True)

        # Extract ranks and suits
        rank1, suit1 = card1[1], str(card1[0])
        rank2, suit2 = card2[1], str(card2[0])

        # Determine hand type
        if rank1 == rank2:
            return rank1 + rank2  # e.g., "AA" for pocket aces
        elif suit1 == suit2:
            return rank1 + rank2 + 's'  # e.g., "AKs" for suited
        else:
            return rank1 + rank2 + 'o'  # e.g., "AKo" for offsuit

    def community_card_to_tuple(self, community_card):
        """
        :param community_card: round_state['community_card']
        :return: tuple of int (0..52)
        """
        new_community_card = []
        for i in range(0, len(community_card)):
            new_community_card.append(self.card_to_int(community_card[i]))
        for i in range(0, 5 - len(community_card)):
            # if community card num <5, append 52 to fill out the rest
            new_community_card.append(52)
        return tuple(new_community_card)

    def eps_greedy_policy(self, s, opponent, valid_actions, eps):
        with torch.no_grad():
            X = torch.tensor([s], device=self.device, dtype=torch.float)
            # Ensure hidden state is correctly formatted for a batch size of 1
            current_hidden = None
            if self.policy_hidden is not None:
                 current_hidden = (self.policy_hidden[0].clone().detach(), self.policy_hidden[1].clone().detach())

            q_values_full, new_hidden = self.policy_net(X, current_hidden)
            self.policy_hidden = new_hidden # Store new hidden state if stateful LSTM is desired hand-wise
            q_values = q_values_full.cpu().numpy().reshape(-1)

            # Create a mask of valid actions
            valid_action_mask = np.ones(self.num_actions, dtype=bool)

            # Adjust mask based on game state
            if opponent['state'] == 'allin' or valid_actions[2]['amount']['max'] == -1:
                valid_action_mask[2:] = False  # Only 'fold' and 'call' are valid
            if valid_actions[1]['amount'] == 0:
                valid_action_mask[0] = False  # Cannot 'fold' when 'check' is available

            # Apply mask to Q-values
            masked_q_values = np.where(valid_action_mask, q_values, -np.inf)

            if np.random.random() >= eps or not self.training:
                action = np.argmax(masked_q_values)
            else:
                valid_actions_indices = np.where(valid_action_mask)[0]
                action = np.random.choice(valid_actions_indices)

            return action

    @staticmethod
    def process_state(s):
        new_s = list(s)
        # Original 46 features:
        # hole_card (0-1), community_card (2-6), main_pot (7), player_stacks (8-13), 
        # self.stack (14), action_history (15-38), player_statuses (39-44), self_position (45)
        
        # Normalize card values (indices 0-6 for 7 cards: 2 hole + 5 community)
        for i in range(0, 7):
            new_s[i] = new_s[i] / 26.0 - 1 # Max card value 51. (51/26.0 - 1) approx 1. (0/26.0 - 1) = -1.
        
        # Normalize pot and stack values
        # main_pot (idx 7), player_stacks (idx 8-13), self.stack (idx 14)
        # total_invested_street_opp_X (part of new features, handled later)
        # effective_stack_size (new feature, handled later)
        for i in range(7, 14 + 1): # Up to index 14 (self.stack)
             new_s[i] = (new_s[i] - 100.0) / 100.0 # Assuming stacks are around 100-200

        # Normalize action_history (bet amounts, indices 15-38)
        for i in range(15, 38 + 1):
            new_s[i] = (new_s[i] - 100.0) / 100.0 # Bet amounts

        # player_statuses (idx 39-44) are 0 or 1, no normalization needed or (val - 0.5) * 2
        # self_position (idx 45) is 0-5 for 6 players, normalize: val / 5.0
        if len(new_s) > 45:
            new_s[45] = new_s[45] / 5.0

        # New features start from index 46
        # num_active_players (idx 46)
        if len(new_s) > 46:
            new_s[46] = new_s[46] / 6.0 
        # effective_stack_size (idx 47)
        if len(new_s) > 47:
            new_s[47] = (new_s[47] - 100.0) / 100.0
        # pot_odds (idx 48) - already 0 to 1
        
        # street_action_history (indices 49 to 49 + 30 - 1 = 78)
        # 5 opponents * 6 features each
        # fold (0/1), check (0/1 or count), call (count), bet (count), raise (count), invested_amount
        action_hist_start_idx = 49
        for opp_idx in range(5):
            base = action_hist_start_idx + opp_idx * 6
            # fold (base + 0), check (base + 1) are 0/1, could be (val - 0.5) * 2 or leave as is
            if len(new_s) > base + 1:
                new_s[base+0] = (new_s[base+0] - 0.5) * 2.0 # map 0,1 to -1,1
                new_s[base+1] = (new_s[base+1] - 0.5) * 2.0 # map 0,1 to -1,1 (if treated as boolean)
                                                            # If counts: new_s[base+1] / 5.0 (max 5 checks?)
            # call, bet, raise counts (base + 2, base + 3, base + 4)
            if len(new_s) > base + 4:
                for i in range(2, 5): # Counts for call, bet, raise
                    new_s[base + i] = new_s[base + i] / 5.0 # Assuming max 5 of each action type
            # total_invested_this_street (base + 5)
            if len(new_s) > base + 5:
                 new_s[base + 5] = (new_s[base + 5] - 100.0) / 100.0 # Normalize around 1BB-2BB investment
        
        return tuple(new_s)
    
    def get_state(self, community_card, round_state):

        # --- Original 46 features ---
        current_state_list = list(self.hole_card + community_card) # 2 + 5 = 7 features

        main_pot_total = round_state['pot']['main']['amount']
        for side_pot in round_state['pot'].get('side', []):
            main_pot_total += side_pot['amount']
        current_state_list.append(main_pot_total) # Feature 8 (main_pot)

        player_stacks_list = [0.0] * 6
        player_statuses_list = [0.0] * 6
        player_uuids = [p['uuid'] for p in round_state['seats']]

        for i in range(6):
            if i < len(round_state['seats']):
                seat = round_state['seats'][i]
                player_stacks_list[i] = seat['stack']
                player_statuses_list[i] = 1.0 if seat['state'] == 'participating' else 0.0
            # else keep as 0 (e.g. if fewer than 6 players registered for some reason)
        
        current_state_list.extend(player_stacks_list) # Features 9-14 (player_stacks)
        current_state_list.append(round_state['seats'][self.player_id]['stack']) # Feature 15 (self.stack) - somewhat redundant

        last_amounts_list = [0.0] * 24 # 6 players * 4 streets
        for street_idx, street_name in enumerate(['preflop', 'flop', 'turn', 'river']):
            actions_on_street = round_state['action_histories'].get(street_name, [])
            for action in actions_on_street:
                try:
                    player_idx_in_game = player_uuids.index(action['uuid']) # Find index based on initial seat order
                    # Map to fixed 0-5 index for last_amounts
                    # This assumes player_uuids indices match 0-5 consistently
                    if 0 <= player_idx_in_game < 6:
                         last_amounts_list[player_idx_in_game * 4 + street_idx] = action.get('amount', 0)
                except ValueError:
                    pass # Player who acted is not in current seats list (should not happen if seats is complete)
        current_state_list.extend(last_amounts_list) # Features 16-39 (action_history)
        current_state_list.extend(player_statuses_list) # Features 40-45 (player_statuses)

        my_seat_idx_in_game = self.player_id # our own seat index
        dealer_btn_seat_idx = round_state['dealer_btn']
        num_game_players = len(round_state['seats'])
        my_position_relative_to_dealer = (my_seat_idx_in_game - dealer_btn_seat_idx + num_game_players) % num_game_players
        current_state_list.append(float(my_position_relative_to_dealer)) # Feature 46 (self_position)
        
        # --- New features (starting from index 46) ---
        
        # 1. Number of active players
        active_players_count = sum(1 for p_status in player_statuses_list if p_status == 1.0)
        current_state_list.append(float(active_players_count)) # Feature 47

        # 2. Effective stack size
        my_current_stack = round_state['seats'][self.player_id]['stack']
        active_player_stacks_values = [player_stacks_list[i] for i, status in enumerate(player_statuses_list) if status == 1.0]
        if not active_player_stacks_values: # If somehow no one is active (e.g. end of hand before processing)
            effective_stack = my_current_stack
        else:
            effective_stack = min(active_player_stacks_values) if active_player_stacks_values else my_current_stack
        current_state_list.append(float(effective_stack)) # Feature 48

        # 3. Pot Odds
        pot_odds = 0.0
        # valid_actions is passed to declare_action, not available here directly.
        # Assuming it would be calculated in declare_action and passed or we use a placeholder
        # For now, let's add placeholder, this needs to be calculated where valid_actions is available
        # typically before calling eps_greedy_policy
        # Let's try to get it from round_state if possible (amount to call for US)
        amount_to_call_for_me = 0
        current_bet_level = 0
        my_bet_this_street = 0
        street_actions = round_state['action_histories'].get(round_state['street'], [])
        if street_actions:
            # Highest bet/raise amount on the street so far
            for act_hist_item in street_actions:
                 current_bet_level = max(current_bet_level, act_hist_item.get('paid', 0)) # 'paid' might be better if available
                                                                                       # or sum of 'amount' if it's additive
            # What I have put in this street
            for act_hist_item in street_actions:
                if act_hist_item['uuid'] == self.uuid:
                    my_bet_this_street += act_hist_item.get('amount',0) # sum my bets if multiple betting rounds on street
            
            amount_to_call_for_me = current_bet_level - my_bet_this_street


        if (main_pot_total + amount_to_call_for_me) > 0 and amount_to_call_for_me > 0:
            pot_odds = amount_to_call_for_me / (main_pot_total + amount_to_call_for_me)
        current_state_list.append(pot_odds) # Feature 49

        # 4. Current street action history for up to 5 opponents (30 features)
        # (fold_count, check_count, call_count, bet_count, raise_count, total_invested_this_street)
        street_action_hist_features = [0.0] * 30 
        current_street_name = round_state['street']
        actions_this_street = round_state['action_histories'].get(current_street_name, [])
        
        opponent_slot = 0
        for i in range(num_game_players):
            if i == self.player_id:
                continue
            if opponent_slot >= 5: # Max 5 opponents
                break

            opp_uuid = round_state['seats'][i]['uuid']
            opp_actions_on_street_summary = {'fold':0, 'check':0, 'call':0, 'bet':0, 'raise':0, 'invested':0}
            
            for action_item in actions_this_street:
                if action_item['uuid'] == opp_uuid:
                    action_type = action_item['action']
                    amount = action_item.get('amount', 0)
                    if action_type == 'fold': opp_actions_on_street_summary['fold'] = 1 # Fold is a state
                    elif action_type == 'call':
                        if amount == 0: opp_actions_on_street_summary['check'] += 1
                        else: opp_actions_on_street_summary['call'] += 1
                    elif action_type == 'raise': opp_actions_on_street_summary['raise'] += 1 # Includes bet if it's first to open
                    # Distinguish bet from raise based on context if necessary, or treat all aggressive open/reraise as 'raise' count
                    # For simplicity, 'raise' covers bets too. Or add 'bet' if first aggressive action.
                    # Let's refine: if it's the first voluntary put money in pot (not blind) its a 'bet'
                    # if it's increasing a previous bet, its a 'raise'

                    # Simplified: using pypokerengine's action types. 'raise' is fine.
                    opp_actions_on_street_summary['invested'] += amount
            
            base_idx = opponent_slot * 6
            street_action_hist_features[base_idx + 0] = float(opp_actions_on_street_summary['fold'])
            street_action_hist_features[base_idx + 1] = float(opp_actions_on_street_summary['check'])
            street_action_hist_features[base_idx + 2] = float(opp_actions_on_street_summary['call'])
            # For bet/raise, we need to be careful. pypokerengine uses 'raise' for any bet beyond a call.
            # Let's assume 'raise' count includes what might be considered 'bet'.
            street_action_hist_features[base_idx + 3] = float(opp_actions_on_street_summary.get('bet',0)) # if we had a separate bet
            street_action_hist_features[base_idx + 4] = float(opp_actions_on_street_summary['raise'])
            street_action_hist_features[base_idx + 5] = float(opp_actions_on_street_summary['invested'])
            
            opponent_slot += 1
            
        current_state_list.extend(street_action_hist_features) # Features 50-79
        
        self.state = tuple(current_state_list)
        return self.state

    def declare_action(self, valid_actions, hole_card, round_state):
        """
        state: hole_card, community_card, self.stack
        """

        # preprocess variable in states
        self.episode += 1
        self.update_count += 1

        hole_card_1 = self.card_to_int(hole_card[0])
        hole_card_2 = self.card_to_int(hole_card[1])
        self.hole_card = (hole_card_1, hole_card_2)
        community_card = self.community_card_to_tuple(round_state['community_card'])
        
        self.state = self.get_state(community_card, round_state)

        self.state = self.process_state(self.state)

        pot_size = round_state['pot']['main']['amount']
        side_pots = round_state['pot'].get('side', [])
        for side_pot in side_pots:
            pot_size += side_pot['amount']

        for action in valid_actions:
            if action['action'] == 'raise':

                min_raise = action['amount']['min']
                max_raise = action['amount']['max']


                for raisesize in self.raisesizes:
            
                    raiseamount = raisesize * pot_size
                    if raiseamount > min_raise and raiseamount < max_raise:
                        action['amount'][raisesize] = raiseamount
                    elif raiseamount <= min_raise: 
                        action['amount'][raisesize] = min_raise
                    elif raiseamount >= max_raise: 
                        action['amount'][raisesize] = max_raise

                amounts = action['amount']
                # Sort the 'min' and 'max'

                min_value = {'min': amounts.pop('min')}
                max_value = {'max': amounts.pop('max')}

                sorted_amount = dict(sorted(amounts.items(), key=lambda item: item[1]))

                final_amount = {**min_value, **sorted_amount, **max_value}
                # Update the original data with sorted values
                action['amount'] = final_amount
        

        action = self.eps_greedy_policy(self.state, round_state['seats'][(self.player_id + 1) % 6], valid_actions,
                                        self.epsilon)
        
        # record the action
        self.history.append(self.state + (action,))

        if action > 1:
            amount = list(valid_actions[2]["amount"].items())[action - 2][1]
            action = "raise"
        elif action == 1:
            amount = valid_actions[1]["amount"]
            action = "call"
        else:
            amount = 0
            action = "fold"

        if round_state["street"] == 'preflop':

            pre_flop_action = round_state['action_histories']['preflop']

            if action in ["call", "raise"]:

                # VPIP Logic
                if amount > 0 and not self.last_vpip_action:
                    self.VPIP += 1
                    self.last_vpip_action = action
    
                # PFR Logic
                if action == "raise" and not self.last_pfr_action:
                    self.PFR += 1
                    self.last_pfr_action = action

                # 3-Bet Logic
                if action == "raise" and not self.last_3_bet_action:
                    # Check for any previous raise in the action history
                    for previous_action in reversed(pre_flop_action[:-1]):  # Go backwards to find the first raise
                        if previous_action["action"].lower() == "raise":
                            self.three_bet += 1
                            self.last_3_bet_action = action
                            break  # Exit loop after the first raise found

        stats_action = action

        if action == "call" and amount == 0:
            stats_action = 'check'
        
        if not self.street_first_action:
            self.action_stat[round_state["street"]][stats_action] += 1 
            self.card_action_stat[round_state["street"]][stats_action][self.hand_type] += 1
            self.street_first_action = stats_action
            if round_state["street"] == 'preflop':
                self.hand_count += 1

        return action, math.floor(amount)

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        for i in range(0, len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i
                break
        self.stack = 100

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.last_vpip_action = None
        self.last_pfr_action = None
        self.last_3_bet_action = None
        self.hand_type = self.get_hand_type(hole_card)
        # Reset LSTM hidden state for the policy network at the start of each hand
        if self.policy_net.lstm_hidden_size > 0: # check if LSTM is part of the model
            self.policy_hidden = (torch.zeros(1, 1, self.policy_net.lstm_hidden_size).to(self.device),
                                  torch.zeros(1, 1, self.policy_net.lstm_hidden_size).to(self.device))
        else:
            self.policy_hidden = None

        pass

    def receive_street_start_message(self, street, round_state):
        self.street_first_action = None
        pass

    def receive_game_update_message(self, new_action, round_state):
        pass

    @staticmethod
    def round_int_to_string(round_int):
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return m[round_int]

    def receive_round_result_message(self, winners, hand_info, round_state):

        if len(self.history) >= 1 and self.training:
            
            for players in round_state['seats']: 
                if players['uuid'] == self.uuid:
                    reward = players['stack'] - self.stack
                    self.stack = players['stack']
                    break

            reward /= 100
            
            self.accumulated_reward += reward

            #if winners[0]['uuid'] == self.uuid:
                #reward += 0.05

            action_num = len(round_state['action_histories'])
            if round_state['action_histories'][self.round_int_to_string(action_num - 1)] == []:
                action_num -= 1
            if action_num == 1:
                community_card = self.community_card_to_tuple([])
            elif action_num == 2:
                community_card = self.community_card_to_tuple(round_state['community_card'][:3])
            elif action_num == 3:
                community_card = self.community_card_to_tuple(round_state['community_card'][:4])
            elif action_num == 4:
                community_card = self.community_card_to_tuple(round_state['community_card'][:5])

            self.state = self.get_state(community_card, round_state)
            last_state = self.process_state(self.state)
            # append the last state to history
            self.history.append(last_state + (None,))

            # update using reward
            for i in range(len(self.history) - 1):
                h = self.history[i]
                next_h = self.history[i + 1]
                self.update(h[:-1], h[-1], reward, next_h[:-1], self.episode)
            # clear history
            self.history = []
            #self.save_model()



            self.card_reward_stat[self.hand_type] += reward



    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_path)