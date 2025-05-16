from pypokerengine.players import BasePokerPlayer
import random as rand
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections import deque, defaultdict

# Hyper-parameters
batch_size = 64
learning_rate_br = 1e-4
learning_rate_avg = 1e-4
gamma = 0.99
br_memory_size = 200000
sl_memory_size = 1000000
# anticipatory_param will be annealed
eta = 0.1  # Learning rate for average policy updates (policy_optimizer already uses learning_rate_avg)
target_net_update_freq = 10000 # Will be replaced by soft updates if polyak_tau is used
epsilon_start = 0.2
epsilon_end = 0.001
epsilon_decay = 500000
min_buffer_size_for_training = 1000

# Max players for one-hot encoding position
MAX_PLAYERS = 6


class ReservoirBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.size = 0
        
    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            idx = rand.randint(0, self.size)
            if idx < self.capacity:
                self.buffer[idx] = transition
        self.size += 1
        
    def sample(self, batch_size):
        return rand.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, transition):
        self.memory.append(transition)
        
    def sample(self, batch_size):
        return rand.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, input_raw_feat_dim, num_actions, lstm_hidden_size=128, card_embedding_dim=16, num_card_indices=7):
        super(QNetwork, self).__init__()
        
        self.num_card_indices = num_card_indices
        self.card_embedding_dim = card_embedding_dim
        self.other_features_dim = input_raw_feat_dim - self.num_card_indices

        self.card_embedding = nn.Embedding(53, self.card_embedding_dim) # 52 cards + 1 padding
        
        lstm_input_dim = (self.num_card_indices * self.card_embedding_dim) + self.other_features_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)
        
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        # Dueling DQN: Value stream
        self.value_fc = nn.Linear(256, 128)
        self.value_ln = nn.LayerNorm(128)
        self.value_output = nn.Linear(128, 1)
        
        # Dueling DQN: Advantage stream
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage_ln = nn.LayerNorm(128)
        self.advantage_output = nn.Linear(128, num_actions)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, num_raw_features)
        card_indices = x[:, :self.num_card_indices].long() # Shape: (batch_size, num_card_indices)
        other_features = x[:, self.num_card_indices:]   # Shape: (batch_size, other_features_dim)
        
        embedded_cards = self.card_embedding(card_indices) # Shape: (batch_size, num_card_indices, card_embedding_dim)
        embedded_cards_flat = embedded_cards.view(embedded_cards.size(0), -1) # Shape: (batch_size, num_card_indices * card_embedding_dim)
        
        combined_features = torch.cat((embedded_cards_flat, other_features), dim=1) # Shape: (batch_size, lstm_input_dim)
        
        if combined_features.ndim == 2: 
            combined_features_unsqueezed = combined_features.unsqueeze(1) # Shape: (batch_size, 1, lstm_input_dim)
        else: # Should already be (batch, seq_len, feature_dim) if called with history
            combined_features_unsqueezed = combined_features

        lstm_out, hidden = self.lstm(combined_features_unsqueezed, hidden)
        lstm_out_last_step = lstm_out[:, -1, :]

        x_fc = F.relu(self.ln1(self.fc1(lstm_out_last_step)))
        x_fc = F.relu(self.ln2(self.fc2(x_fc)))
        
        value = F.relu(self.value_ln(self.value_fc(x_fc)))
        value = self.value_output(value) # Shape: (batch_size, 1)
        
        advantage = F.relu(self.advantage_ln(self.advantage_fc(x_fc)))
        advantage = self.advantage_output(advantage) # Shape: (batch_size, num_actions)
        
        # Combine value and advantage: Q(s,a) = V(s) + (A(s,a) - mean_a(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values, hidden

class PolicyNetwork(nn.Module):
    def __init__(self, input_raw_feat_dim, num_actions, lstm_hidden_size=128, card_embedding_dim=16, num_card_indices=7):
        super(PolicyNetwork, self).__init__()

        self.num_card_indices = num_card_indices
        self.card_embedding_dim = card_embedding_dim
        self.other_features_dim = input_raw_feat_dim - self.num_card_indices

        self.card_embedding = nn.Embedding(53, self.card_embedding_dim)

        lstm_input_dim = (self.num_card_indices * self.card_embedding_dim) + self.other_features_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)

        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, num_actions)
        
    def forward(self, x, hidden=None):
        card_indices = x[:, :self.num_card_indices].long()
        other_features = x[:, self.num_card_indices:]
        
        embedded_cards = self.card_embedding(card_indices)
        embedded_cards_flat = embedded_cards.view(embedded_cards.size(0), -1)
        
        combined_features = torch.cat((embedded_cards_flat, other_features), dim=1)
        
        if combined_features.ndim == 2:
            combined_features_unsqueezed = combined_features.unsqueeze(1)
        else:
            combined_features_unsqueezed = combined_features

        lstm_out, hidden = self.lstm(combined_features_unsqueezed, hidden)
        lstm_out_last_step = lstm_out[:, -1, :]

        x_fc = F.relu(self.ln1(self.fc1(lstm_out_last_step)))
        x_fc = F.relu(self.ln2(self.fc2(x_fc)))
        x_fc = F.relu(self.ln3(self.fc3(x_fc)))
        x_fc = F.softmax(self.fc4(x_fc), dim=1)
        return x_fc, hidden

class NFSPPlayer(BasePokerPlayer):
    def __init__(self, 
                 shared_q_net, shared_target_q_net, shared_policy_net,
                 shared_q_optimizer, shared_policy_optimizer,
                 q_model_save_path, policy_model_save_path,
                 q_optimizer_save_path, policy_optimizer_save_path,
                 training=True,
                 card_embedding_dim=16, sl_update_freq=1, br_update_freq=1, polyak_tau=0.005,
                 anticipatory_param_start=0.1, anticipatory_param_end=0.1, anticipatory_param_decay=1000000,
                 # Network/Player Structure parameters, still needed for player logic and for training.py to init shared nets
                 num_feats=85, num_actions=11, lstm_hidden_size=128, num_card_indices_in_state=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_player = None
        self.player_id = None
        self.episode = 0 # Game count
        
        self.br_memory = ReplayMemory(br_memory_size)
        self.sl_memory = ReservoirBuffer(sl_memory_size)
        
        self.q_loss = None
        self.policy_loss = None
        
        self.gamma = gamma
        
        # Annealable anticipatory parameter
        self.anticipatory_param_start = anticipatory_param_start
        self.anticipatory_param_end = anticipatory_param_end
        self.anticipatory_param_decay = anticipatory_param_decay
        self.anticipatory_param = anticipatory_param_start
        
        self.batch_size = batch_size # Global batch_size
        
        self.epsilon = epsilon_start # Global epsilon_start, initialize current epsilon
        self.epsilon_start = epsilon_start # Store for decay calculation
        self.epsilon_end = epsilon_end     # Store for decay calculation
        self.epsilon_decay = epsilon_decay   # Store for decay calculation
        
        self.stack = 100
        self.hole_card_tuple_raw_indices = None # Store raw card indices for heuristic
        
        # Store shared paths
        self.q_model_save_path = q_model_save_path
        self.policy_model_save_path = policy_model_save_path
        self.q_optimizer_save_path = q_optimizer_save_path
        self.policy_optimizer_save_path = policy_optimizer_save_path
        
        self.update_count = 0 # Number of times train_networks is called
        self.training = training
        self.best_response_mode = None
        self.current_state = None
        self.history = []
        self.lstm_hidden_size = lstm_hidden_size
        self.q_net_hidden = None
        self.policy_net_hidden = None
        
        self.hand_count = 0
        self.VPIP = 0
        self.last_vpip_action = None
        self.PFR = 0
        self.last_pfr_action = None
        self.three_bet = 0
        self.last_3_bet_action = None
        self.street_first_action = None
        
        self.raisesizes = [0.33, 0.5, 0.75, 1, 1.5, 2.0, 2.5, 3.0]
        
        self.last_reward = 0
        self.accumulated_reward = 0
        
        self.hand_strength_cache = {}
        self.action_stat = {
            'preflop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'flop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'turn': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'river': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0}
        }
        self.hand_type_str = None # For stat tracking like card_reward_stat
        self.card_reward_stat = self.initialize_card_stats()
        self.card_action_stat = self.initialize_card_action_stats()
        
        self.num_actions = num_actions
        self.num_feats = num_feats 
        self.num_card_indices_in_state = num_card_indices_in_state 
        self.card_embedding_dim = card_embedding_dim

        # For soft target updates
        self.polyak_tau = polyak_tau
        # For decoupled updates
        self.sl_update_freq = sl_update_freq
        self.br_update_freq = br_update_freq

        # Assign shared networks and optimizers directly
        self.q_net = shared_q_net
        self.target_q_net = shared_target_q_net
        self.policy_net = shared_policy_net
        self.q_optimizer = shared_q_optimizer
        self.policy_optimizer = shared_policy_optimizer

        # Ensure networks are in the correct mode (train/eval) based on self.training
        # This should be done after they are assigned.
        if self.training:
            if self.q_net: self.q_net.train()
            if self.target_q_net: self.target_q_net.train() # Typically target_q_net mirrors q_net mode or is eval
            if self.policy_net: self.policy_net.train()
        else:
            if self.q_net: self.q_net.eval()
            if self.target_q_net: self.target_q_net.eval()
            if self.policy_net: self.policy_net.eval()
        
        # Initial sync of target_q_net with q_net is crucial and should happen once
        # when shared_target_q_net is created/loaded in training.py.
        # If self.target_q_net.load_state_dict(self.q_net.state_dict()) was here,
        # it would cause issues if called by multiple agents on already synced shared nets.
        # It's better handled centrally in training.py after shared q_net and target_q_net are ready.
        print("NFSPPlayer: Using pre-initialized shared networks and optimizers.")

    def initialize_card_stats(self): # For hand_type_str based stats
        card_stats = {}
        # ranks_str is ordered from highest rank (A) to lowest (2) for easier iteration
        ranks_str_ordered = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2'] 
        
        for r1_char_idx, r1_char in enumerate(ranks_str_ordered):
            # Pairs
            card_stats[r1_char + r1_char] = 0
            # Suited and Offsuit (r1_char is always higher or equal rank than r2_char here)
            for r2_char_idx in range(r1_char_idx + 1, len(ranks_str_ordered)):
                r2_char = ranks_str_ordered[r2_char_idx]
                # r1_char is guaranteed to be higher rank than r2_char due to loop structure
                card_stats[r1_char + r2_char + 's'] = 0 # e.g. AKs, A2s, KQs, 72s
                card_stats[r1_char + r2_char + 'o'] = 0 # e.g. AKo, A2o, KQo, 72o
        return card_stats
    
    def initialize_card_action_stats(self):
        card_action_stats = {
            'preflop': {}, 'flop': {}, 'turn': {}, 'river': {}
        }
        for street in card_action_stats.keys():
            card_action_stats[street]['raise'] = self.initialize_card_stats()
            card_action_stats[street]['call'] = self.initialize_card_stats()
            card_action_stats[street]['check'] = self.initialize_card_stats()
            card_action_stats[street]['fold'] = self.initialize_card_stats()
        return card_action_stats
    
    def initialize_networks(self):
        # This method is now simplified when using shared networks.
        # The actual creation and loading of shared networks happens in training.py.
        # This player instance receives already initialized (and possibly pre-loaded) shared networks.
        # Ensure the networks are set to the correct device and mode (train/eval).

        # Networks and optimizers are already assigned in __init__ from shared instances.
        # The device transfer (.to(self.device)) for shared networks should happen in training.py 
        # when the shared networks are first created.

        # Set training/eval mode based on self.training status
        if self.training:
            if self.q_net: self.q_net.train()
            if self.target_q_net: self.target_q_net.train() # Or eval, depending on DQN target net practices
            if self.policy_net: self.policy_net.train()
        else:
            if self.q_net: self.q_net.eval()
            if self.target_q_net: self.target_q_net.eval()
            if self.policy_net: self.policy_net.eval()

        # Initial sync of target_q_net with q_net is crucial and should happen once
        # when shared_target_q_net is created/loaded in training.py.
        # If self.target_q_net.load_state_dict(self.q_net.state_dict()) was here,
        # it would cause issues if called by multiple agents on already synced shared nets.
        # It's better handled centrally in training.py after shared q_net and target_q_net are ready.
        print("NFSPPlayer: Using pre-initialized shared networks and optimizers.")

    def save_model(self):
        # Save the shared networks and optimizers using the shared paths
        if self.q_net: torch.save(self.q_net.state_dict(), self.q_model_save_path)
        if self.policy_net: torch.save(self.policy_net.state_dict(), self.policy_model_save_path)
        if self.q_optimizer: torch.save(self.q_optimizer.state_dict(), self.q_optimizer_save_path)
        if self.policy_optimizer: torch.save(self.policy_optimizer.state_dict(), self.policy_optimizer_save_path)
        # print(f"NFSP Shared Models saved to {self.q_model_save_path}, {self.policy_model_save_path}") # Optional: more detailed print

    def update_target_network(self):
        # Soft update
        if self.polyak_tau is not None:
            for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.polyak_tau * local_param.data + (1.0 - self.polyak_tau) * target_param.data)
        # Hard update (if polyak_tau is None or for compatibility)
        elif self.update_count % target_net_update_freq == 0: # Original global target_net_update_freq
             self.target_q_net.load_state_dict(self.q_net.state_dict())

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.update_count / self.epsilon_decay) # Use update_count for decay

    def update_anticipatory_param(self):
        if self.anticipatory_param_start != self.anticipatory_param_end : # Only anneal if start and end are different
            self.anticipatory_param = self.anticipatory_param_end + \
                                  (self.anticipatory_param_start - self.anticipatory_param_end) * \
                                  math.exp(-1. * self.update_count / self.anticipatory_param_decay)


    def select_action(self, state, valid_actions):
        if self.best_response_mode is None: # Decide mode once per hand/episode start
            self.best_response_mode = np.random.random() < self.anticipatory_param
        
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=self.device, dtype=torch.float)
            valid_action_mask = np.ones(self.num_actions, dtype=bool)
            if valid_actions[1]['amount'] == 0:
                valid_action_mask[0] = False
            if len(valid_actions) < 3 or valid_actions[2]['amount']['max'] == -1:
                valid_action_mask[2:] = False
            else:
                # This logic assumes valid_actions[2]['amount'] has distinct raise amounts excluding min/max
                # The keys in valid_actions[2]['amount'] might be like 'raise_option_0', 'raise_option_1'...
                # Or they might be the actual float values as strings like "0.5", "1.0" from declare_action.
                # Let's count items that are not 'min' or 'max'.
                num_available_raise_options = sum(1 for k in valid_actions[2]['amount'] if k not in ['min', 'max'])
                for k_raise_slot in range(self.num_actions - 2): 
                    if k_raise_slot >= num_available_raise_options:
                        valid_action_mask[2 + k_raise_slot] = False
            
            action = 0 # Initialize action
            if self.best_response_mode:
                current_q_hidden = self.q_net_hidden # Use stored hidden state
                q_values_full, new_q_hidden = self.q_net(state_tensor, current_q_hidden)
                self.q_net_hidden = new_q_hidden 
                q_values = q_values_full.cpu().numpy().reshape(-1)
                masked_q_values = np.where(valid_action_mask, q_values, -np.inf)
                if np.random.random() < self.epsilon and self.training:
                    valid_indices = np.where(valid_action_mask)[0]
                    action = np.random.choice(valid_indices) if len(valid_indices) > 0 else np.argmax(masked_q_values) # Fallback if no valid random choice
                else:
                    action = np.argmax(masked_q_values)
            else:
                current_policy_hidden = self.policy_net_hidden # Use stored hidden state
                policy_probs_full, new_policy_hidden = self.policy_net(state_tensor, current_policy_hidden)
                self.policy_net_hidden = new_policy_hidden
                policy_probs = policy_probs_full.cpu().numpy().reshape(-1)
                masked_policy_probs = np.where(valid_action_mask, policy_probs, 0)
                if np.sum(masked_policy_probs) > 0:
                    masked_policy_probs = masked_policy_probs / np.sum(masked_policy_probs)
                else:
                    # Fallback: uniform distribution over valid actions if all masked probs are 0
                    valid_indices = np.where(valid_action_mask)[0]
                    if len(valid_indices) > 0:
                        masked_policy_probs = np.zeros_like(policy_probs)
                        masked_policy_probs[valid_indices] = 1.0 / len(valid_indices)
                    else: # Should not happen if valid_actions always has at least one option (fold or call)
                        masked_policy_probs[0] = 1.0 # Default to fold if truly no valid action
                
                action = np.random.choice(self.num_actions, p=masked_policy_probs)
                
            if self.best_response_mode and self.training: # Store (state, BR_action) for SL
                self.sl_memory.push((state, action))
            return action
    
    def store_transition(self, state, action, reward, next_state):
        # This is for the BR (Q-learning) part
        if self.training and self.best_response_mode: # Only store if BR was active for this hand
            self.br_memory.push((state, action, reward, next_state))
    
    def train_q_network(self):
        if len(self.br_memory) < min_buffer_size_for_training:
            return
        transitions = self.br_memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        state_batch = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch_action, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch_reward, device=self.device, dtype=torch.float).unsqueeze(1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), 
                                    device=self.device, dtype=torch.bool)
        non_final_next_states_list = [s for s in batch_next_state if s is not None]
        
        q_values_full, _ = self.q_net(state_batch) # No hidden state passed for batch training
        q_values = q_values_full.gather(1, action_batch)
        
        next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
        if len(non_final_next_states_list) > 0:
            non_final_next_states_tensor = torch.tensor(np.array(non_final_next_states_list), device=self.device, dtype=torch.float)
            target_q_full, _ = self.target_q_net(non_final_next_states_tensor) # No hidden state
            next_q_values[non_final_mask] = target_q_full.max(1, keepdim=True)[0].detach()
        
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.q_loss = loss.item()
        self.q_optimizer.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.q_optimizer.step()
    
    def train_policy_network(self):
        if len(self.sl_memory) < min_buffer_size_for_training:
            return
        transitions = self.sl_memory.sample(self.batch_size)
        batch_state, batch_action = zip(*transitions)
        state_batch = torch.tensor(np.array(batch_state), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch_action, device=self.device, dtype=torch.long)
        
        policy_probs_full, _ = self.policy_net(state_batch) # No hidden state passed for batch training
        log_probs = torch.log(policy_probs_full + 1e-8) 
        loss = F.nll_loss(log_probs, action_batch)
        self.policy_loss = loss.item()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
    
    def update(self): # Removed episode arg, use self.episode or self.update_count
        if not self.training:
            return
        
        if self.update_count % self.br_update_freq == 0:
            self.train_q_network()
        if self.update_count % self.sl_update_freq == 0:
            self.train_policy_network()
        
        self.update_target_network() # Soft update happens here if polyak_tau is set
        self.update_epsilon()
        self.update_anticipatory_param() # Anneal anticipatory param
        self.update_count += 1
    
    @staticmethod
    def card_to_int(card): # card format from pypokerengine: "H1" (suit, rank_char)
        suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3} # Or consistent with your card encoding
        rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
        # For embedding, want unique int from 0-51. Max rank is 12.
        # Example: card_idx = suit_val * 13 + rank_val
        suit = card[0]
        rank = card[1]
        if rank == '1': # Handle '10'
            rank = 'T'
            if len(card) == 3 and card[2] == '0': # "S10"
                 pass # rank is already T
            else: # "S1" this should not happen with pypokerengine
                # Fallback or error, for now, assume 'T' if '1' is found.
                pass


        return suit_map[suit] * 13 + rank_map[rank]
    
    def get_hand_type_str(self, hole_card_pokerlib_format): # hole_card_pokerlib_format: ["H1", "S1"]
        # This is for stat tracking with string keys like "AA", "AKs"
        ranks_map_to_char = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
        
        card1_idx = self.card_to_int(hole_card_pokerlib_format[0])
        card2_idx = self.card_to_int(hole_card_pokerlib_format[1])

        rank1_val = card1_idx % 13
        rank2_val = card2_idx % 13
        suit1_val = card1_idx // 13
        suit2_val = card2_idx // 13

        r1_char = ranks_map_to_char[rank1_val]
        r2_char = ranks_map_to_char[rank2_val]

        # Order by rank (higher rank first)
        if rank1_val < rank2_val:
            r1_char, r2_char = r2_char, r1_char
        
        if r1_char == r2_char:
            return r1_char + r2_char
        elif suit1_val == suit2_val:
            return r1_char + r2_char + 's'
        else:
            return r1_char + r2_char + 'o'
    
    def community_card_to_tuple_indices(self, community_card_pokerlib):
        new_community_card_indices = []
        for card_str in community_card_pokerlib:
            new_community_card_indices.append(self.card_to_int(card_str))
        # Pad with 52 (index for padding token for embedding layer)
        while len(new_community_card_indices) < 5:
            new_community_card_indices.append(52) 
        return tuple(new_community_card_indices)
    
    @staticmethod
    def process_state(s_tuple): # s_tuple is raw state from get_state
        s_list = list(s_tuple)
        
        # Normalize features
        # s_list[7] = Pot size: (pot - 200) / 200 (approx avg pot 200, range 0-many)
        s_list[7] = (s_list[7] - 200.0) / 200.0
        
        # s_list[8-13] = Player stacks (6 features): (stack - 100) / 50 (approx avg stack 100, range 0-200+)
        for i in range(8, 14):
            s_list[i] = (s_list[i] - 100.0) / 50.0
        
        # s_list[14] = Self stack: (stack - 100) / 50
        s_list[14] = (s_list[14] - 100.0) / 50.0
        
        # s_list[15-38] = last_amounts (24 features, 6 players * 4 streets): (amount - 50) / 50 (approx avg bet 50)
        for i in range(15, 39):
            s_list[i] = (s_list[i] - 50.0) / 50.0

        # s_list[39-44] = player_statuses_list (6 features): already 0 or 1

        # s_list[7+38] (index 45 of s_list) = my_position_relative_to_dealer (scalar 0-5)
        # Convert scalar position (index 45 in original s_list, which is s_list[7+38]) to one-hot
        pos_scalar_original_idx = 7 + 38 
        if pos_scalar_original_idx < len(s_list):
            pos_scalar = int(s_list[pos_scalar_original_idx])
            pos_one_hot = [0.0] * MAX_PLAYERS
            if 0 <= pos_scalar < MAX_PLAYERS:
                pos_one_hot[pos_scalar] = 1.0
            # Replace scalar position with one-hot encoding
            s_list = s_list[:pos_scalar_original_idx] + pos_one_hot + s_list[pos_scalar_original_idx+1:]
        
        # After one-hot encoding, the features from this point onwards are shifted.
        # Original indices for SPR, pot_odds, active_players, eff_stack, heuristic_strength were:
        # 7+38+1 (SPR), 7+38+2 (PotOdds), 7+38+3 (ActivePlayers), 7+38+4 (EffStack), 7+38+5 (Heuristic)
        # New base index after pos_scalar (1) is replaced by pos_one_hot (6) -> net +5 shift
        base_idx_after_pos = pos_scalar_original_idx + MAX_PLAYERS # Start of features after one-hot position

        # s_list[base_idx_after_pos + 0] = SPR: (spr - 10) / 10 (common SPR around 10)
        if base_idx_after_pos < len(s_list):
            s_list[base_idx_after_pos] = (s_list[base_idx_after_pos] - 10.0) / 10.0
        
        # s_list[base_idx_after_pos + 1] = Pot odds: (odds - 0.33) / 0.15 (typical odds around 0.2-0.5)
        if base_idx_after_pos + 1 < len(s_list):
            s_list[base_idx_after_pos + 1] = (s_list[base_idx_after_pos + 1] - 0.33) / 0.15

        # s_list[base_idx_after_pos + 2] = Active players count: (count - 3) / 2 (avg 2-4 players)
        if base_idx_after_pos + 2 < len(s_list):
             s_list[base_idx_after_pos + 2] = (s_list[base_idx_after_pos + 2] - 3.0) / 2.0
        
        # s_list[base_idx_after_pos + 3] = Effective stack: (eff_stack - 100) / 50
        if base_idx_after_pos + 3 < len(s_list):
            s_list[base_idx_after_pos + 3] = (s_list[base_idx_after_pos + 3] - 100.0) / 50.0
        
        # Heuristic strength was at base_idx_after_pos + 4, now removed.
        # Normalization for street_action_hist_features (opponent actions)
        # These start after the features above. If heuristic was removed, they are now at base_idx_after_pos + 4
        street_hist_start_idx = base_idx_after_pos + 4 

        num_opp_street_hist_sets = 5 # Max 5 opponents
        features_per_opp_street_hist = 6 # fold, check, call, bet, raise, invested
        for opp_set in range(num_opp_street_hist_sets):
            base_street_hist_idx = street_hist_start_idx + (opp_set * features_per_opp_street_hist)
            if len(s_list) > base_street_hist_idx + 4: # call, bet, raise counts
                for i in range(2, 5): 
                    s_list[base_street_hist_idx + i] = s_list[base_street_hist_idx + i] / 3.0 # Normalize counts (e.g. max 3 actions)
            if len(s_list) > base_street_hist_idx + 5: # total_invested_this_street
                s_list[base_street_hist_idx + 5] = (s_list[base_street_hist_idx + 5] - 100.0) / 100.0 # Normalize investment
        
        # Expected final size: 7(cards) + 38(pot,stacks,hist,status) + 6(pos_onehot) + 4(spr,odds,active,eff) + 30(street_hist_opp) = 85
        return tuple(s_list)
    
    def get_state(self, community_card_indices_tuple, round_state): # community_card_indices_tuple are raw indices
        # Hole cards are already stored in self.hole_card_tuple_raw_indices
        current_state_list = list(self.hole_card_tuple_raw_indices + community_card_indices_tuple) # 7 raw card indices

        main_pot_total = round_state['pot']['main']['amount']
        for side_pot in round_state['pot'].get('side', []):
            main_pot_total += side_pot['amount']
        current_state_list.append(main_pot_total) # Feature 7

        player_stacks_list = [0.0] * MAX_PLAYERS
        player_statuses_list = [0.0] * MAX_PLAYERS
        player_uuids = [p['uuid'] for p in round_state['seats']]

        for i in range(MAX_PLAYERS):
            if i < len(round_state['seats']):
                seat = round_state['seats'][i]
                player_stacks_list[i] = seat['stack']
                player_statuses_list[i] = 1.0 if seat['state'] == 'participating' else 0.0
        current_state_list.extend(player_stacks_list) # Features 8-13 (using MAX_PLAYERS size)
        current_state_list.append(round_state['seats'][self.player_id]['stack']) # Feature 14

        last_amounts_list = [0.0] * (MAX_PLAYERS * 4) # MAX_PLAYERS players * 4 streets
        for street_idx, street_name in enumerate(['preflop', 'flop', 'turn', 'river']):
            actions_on_street = round_state['action_histories'].get(street_name, [])
            for action in actions_on_street:
                try:
                    player_idx_in_game = player_uuids.index(action['uuid']) # This is index in current game's seats
                    # We need to map this to the global 0-5 player index if round_state['seats'] can be smaller
                    # For now, assume player_idx_in_game is the global 0-MAX_PLAYERS-1 index
                    if 0 <= player_idx_in_game < MAX_PLAYERS:
                         last_amounts_list[player_idx_in_game * 4 + street_idx] = action.get('amount', 0)
                except ValueError:
                    pass # Player not in current round_state.seats (e.g. if player folded and removed)
        current_state_list.extend(last_amounts_list) # Features 15-38 (24 for 6 players)
        current_state_list.extend(player_statuses_list) # Features 39-44

        my_seat_idx_in_game = self.player_id # This is the agent's index in the round_state.seats
        dealer_btn_seat_idx = round_state['dealer_btn'] # Index of dealer in round_state.seats
        num_game_players = len(round_state['seats'])
        # Position relative to dealer (0=dealer, 1=SB, etc. in the current game setup)
        my_position_relative_to_dealer = (my_seat_idx_in_game - dealer_btn_seat_idx + num_game_players) % num_game_players
        current_state_list.append(float(my_position_relative_to_dealer)) # Feature 45 (scalar position 0-5 for one-hot encoding later)
        
        my_current_stack = round_state['seats'][self.player_id]['stack']
        spr_value = my_current_stack / max(1.0, main_pot_total)
        current_state_list.append(spr_value) # Feature 46

        pot_odds_value = 0.0
        amount_to_call_for_me = 0
        current_bet_level_on_street = 0
        my_bet_this_street = 0
        street_actions = round_state['action_histories'].get(round_state['street'], [])
        if street_actions:
            # Max bet amount on the street so far
            for act_hist_item in street_actions:
                action_total_bet = act_hist_item.get('amount', 0)
                if act_hist_item['action'] == 'raise': # pypokerengine stores 'add_amount' for raises
                    action_total_bet += act_hist_item.get('add_amount', 0)
                current_bet_level_on_street = max(current_bet_level_on_street, action_total_bet)

            # My total investment this street
            for act_hist_item in street_actions:
                if act_hist_item['uuid'] == self.uuid:
                    # Sum up all 'amount' for call/bet and 'amount' + 'add_amount' for raise
                    my_bet_this_street += act_hist_item.get('amount',0)
                    if act_hist_item['action'] == 'raise':
                        my_bet_this_street += act_hist_item.get('add_amount',0)

            amount_to_call_for_me = current_bet_level_on_street - my_bet_this_street
        
        if (main_pot_total + amount_to_call_for_me) > 0 and amount_to_call_for_me > 0:
            pot_odds_value = amount_to_call_for_me / (main_pot_total + amount_to_call_for_me)
        current_state_list.append(pot_odds_value) # Feature 47

        active_players_count = sum(1 for p_status in player_statuses_list if p_status == 1.0)
        current_state_list.append(float(active_players_count)) # Feature 48

        active_player_stacks_values = [player_stacks_list[i] for i, status in enumerate(player_statuses_list) if status == 1.0 and i < len(player_stacks_list)]
        if not active_player_stacks_values: # If only self is active or no one active
            effective_stack = my_current_stack
        else:
            effective_stack = min(active_player_stacks_values) if my_current_stack not in active_player_stacks_values else min(my_current_stack, min(s for s in active_player_stacks_values if s != my_current_stack) if any(s != my_current_stack for s in active_player_stacks_values) else my_current_stack)

        current_state_list.append(float(effective_stack)) # Feature 49
        
        # Street Action History for Opponents
        street_action_hist_features = [0.0] * (5 * 6) # Max 5 opponents, 6 features each
        current_street_name = round_state['street']
        actions_this_street = round_state['action_histories'].get(current_street_name, [])
        
        opponent_slot_street_hist = 0
        for i in range(num_game_players): # Iterate through players in current game
            if i == self.player_id:
                continue # Skip self
            if opponent_slot_street_hist >= 5: 
                break # Max 5 opponents tracked for this feature

            opp_uuid_in_game = round_state['seats'][i]['uuid']
            opp_street_actions = {'fold':0.0, 'check':0.0, 'call':0.0, 'bet':0.0, 'raise':0.0, 'invested':0.0}
            
            for action_item in actions_this_street:
                if action_item['uuid'] == opp_uuid_in_game:
                    action_type = action_item['action']
                    amount = action_item.get('amount', 0)
                    add_amount = action_item.get('add_amount',0) if action_type == 'raise' else 0

                    if action_type == 'fold': opp_street_actions['fold'] = 1.0
                    elif action_type == 'call':
                        if amount == 0: opp_street_actions['check'] += 1.0
                        else: opp_street_actions['call'] += 1.0
                    elif action_type == 'raise': 
                        opp_street_actions['raise'] += 1.0 # Could also try to distinguish bet vs raise based on context
                    
                    opp_street_actions['invested'] += (amount + add_amount)
            
            base_idx = opponent_slot_street_hist * 6
            street_action_hist_features[base_idx + 0] = opp_street_actions['fold']
            street_action_hist_features[base_idx + 1] = opp_street_actions['check']
            street_action_hist_features[base_idx + 2] = opp_street_actions['call']
            street_action_hist_features[base_idx + 3] = opp_street_actions['bet'] # Simple count, refine if needed
            street_action_hist_features[base_idx + 4] = opp_street_actions['raise']
            street_action_hist_features[base_idx + 5] = opp_street_actions['invested']
            
            opponent_slot_street_hist += 1
        current_state_list.extend(street_action_hist_features) # Features 50-79 (30 features) if heuristic was feature 49 + 1
                    
        # Count: 7(cards) + 1(pot) + 6(stacks) + 1(self_stack) + 24(last_amounts) + 6(statuses) + 1(my_pos_scalar) + 1(SPR) + 1(pot_odds) + 1(active_players) + 1(eff_stack) + 30(street_hist_opp) = 80
        return tuple(current_state_list) 
        
    def declare_action(self, valid_actions, hole_card_pokerlib, round_state):
        self.episode +=1 # If self.episode is per hand start, this is fine. If per action, move to round_start.
        
        # Convert hole_card to raw indices and store for heuristic
        hole_card_idx1 = self.card_to_int(hole_card_pokerlib[0])
        hole_card_idx2 = self.card_to_int(hole_card_pokerlib[1])
        self.hole_card_tuple_raw_indices = (hole_card_idx1, hole_card_idx2)
        
        community_card_indices_tuple = self.community_card_to_tuple_indices(round_state['community_card'])
        
        raw_state_tuple = self.get_state(community_card_indices_tuple, round_state)
        processed_state_for_nn = self.process_state(raw_state_tuple) 
        self.current_state = processed_state_for_nn 

        pot_size = round_state['pot']['main']['amount']
        for side_pot in round_state['pot'].get('side', []): pot_size += side_pot['amount']
        my_stack = round_state['seats'][self.player_id]['stack']
        spr = my_stack / max(1, pot_size)


        # Ensure valid_actions[2]['amount'] is structured for select_action mapping
        # It needs 'min', 'max', and then a number of distinct raise options.
        for action_obj in valid_actions:
            if action_obj['action'] == 'raise':
                min_r, max_r = action_obj['amount']['min'], action_obj['amount']['max']
                generated_raises = {}
                for r_size_factor in self.raisesizes:
                    val = r_size_factor * pot_size
                    if min_r != -1 and val < min_r : val = min_r # Clamp to min_raise if valid
                    if max_r != -1 and val > max_r : val = max_r # Clamp to max_raise if valid
                    # Only add if it makes sense (e.g., not less than min, not more than max if they are set)
                    if (min_r == -1 or val >= min_r) and (max_r == -1 or val <= max_r):
                         generated_raises[str(r_size_factor)] = math.floor(val)

                # Filter to unique amounts and sort, then take up to num_actions - 2
                distinct_raise_amounts = sorted(list(set(g_val for g_val in generated_raises.values() if g_val > 0 and (min_r == -1 or g_val >= min_r) and (max_r == -1 or g_val <=max_r) )))
                
                action_obj['amount'] = {'min': min_r, 'max': max_r} # Reset and repopulate
                for i, distinct_amount in enumerate(distinct_raise_amounts[:self.num_actions - 2]):
                    action_obj['amount'][f'raise_opt_{i}'] = distinct_amount


        action_index = self.select_action(processed_state_for_nn, valid_actions)
        action_str, amount = "fold", 0 # Defaults
        if action_index == 1: # Call/Check
            action_str = "call"
            amount = valid_actions[1]["amount"]
        elif action_index > 1: # Raise
            action_str = "raise"
            # Map action_index (2 to N_ACTIONS-1) to the available distinct raise amounts
            # The keys in valid_actions[2]['amount'] should be like 'raise_opt_0', 'raise_opt_1'...
            raise_option_key = f'raise_opt_{action_index - 2}'
            if raise_option_key in valid_actions[2]['amount']:
                amount = valid_actions[2]['amount'][raise_option_key]
            elif valid_actions[2]['amount']['min'] != -1 : # Fallback to min raise if specific option not found but min is possible
                amount = valid_actions[2]['amount']['min']
            elif valid_actions[1]['amount'] != -1: # Fallback to call if raise is not feasible
                 action_str = "call"
                 amount = valid_actions[1]["amount"]
            # If all else fails, it might become a fold if amount remains 0 for raise (should be caught by mask)

        if round_state["street"] == 'preflop':
            # ... (VPIP, PFR, 3-Bet logic - uses action_str and amount) ...
            pre_flop_actions_history = round_state['action_histories']['preflop']
            if action_str in ["call", "raise"]:
                if amount > 0 and not self.last_vpip_action: # VPIP: any voluntary money in pot
                    self.VPIP += 1
                    self.last_vpip_action = action_str 
                if action_str == "raise" and not self.last_pfr_action: # PFR: first raise preflop
                    self.PFR += 1
                    self.last_pfr_action = action_str
                if action_str == "raise" and not self.last_3_bet_action: # 3-Bet: a reraise
                    is_3bet = False
                    raises_in_history = [act for act in pre_flop_actions_history if act['action'] == 'RAISE'] # You must capitalize the action
                    if len(raises_in_history) >= 1: # There was at least one prior raise
                        is_3bet = True
                    if is_3bet:
                        self.three_bet += 1
                        self.last_3_bet_action = action_str
            
        stats_action_to_record = 'check' if action_str == "call" and amount == 0 else action_str
        if not self.street_first_action: # Record first action on street for analysis
            self.action_stat[round_state["street"]][stats_action_to_record] += 1 
            if self.hand_type_str: # Ensure hand_type_str is set
                 self.card_action_stat[round_state["street"]][stats_action_to_record][self.hand_type_str] += 1
            self.street_first_action = stats_action_to_record
            if round_state["street"] == 'preflop':
                self.hand_count += 1
                
        self.history.append((processed_state_for_nn, action_index))
        return action_str, math.floor(amount)
    
    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        for i in range(len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i; break
        self.stack = game_info['seats'][self.player_id]['stack'] # Initial stack
    
    def receive_round_start_message(self, round_count, hole_card_pokerlib, seats):
        self.last_vpip_action = None
        self.last_pfr_action = None
        self.last_3_bet_action = None
        self.hand_type_str = self.get_hand_type_str(hole_card_pokerlib) # Use string for stat keys
        self.best_response_mode = None 
        self.history = []
        
        # Reset LSTM hidden states for new hand sequence
        if self.lstm_hidden_size > 0:
            self.q_net_hidden = (torch.zeros(1, 1, self.lstm_hidden_size).to(self.device),
                                 torch.zeros(1, 1, self.lstm_hidden_size).to(self.device))
            self.policy_net_hidden = (torch.zeros(1, 1, self.lstm_hidden_size).to(self.device),
                                      torch.zeros(1, 1, self.lstm_hidden_size).to(self.device))
        else: # Should not happen if lstm_hidden_size > 0 as per condition
            self.q_net_hidden = None
            self.policy_net_hidden = None
    
    def receive_street_start_message(self, street, round_state):
        self.street_first_action = None
    
    def receive_game_update_message(self, new_action, round_state):
        pass # NFSP typically learns at end of round
    
    @staticmethod
    def round_int_to_string(round_int): # Helper if needed
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river', 4: 'showdown'} # Max 4 betting rounds
        return m.get(round_int, 'unknown')
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        if not self.training or not self.history:
            return
            
        final_stack = 0
        for player_seat_info in round_state['seats']: 
            if player_seat_info['uuid'] == self.uuid:
                final_stack = player_seat_info['stack']
                break
        
        reward = (final_stack - self.stack) / 100.0 # Normalize reward by initial stack or BBs
        self.stack = final_stack # Update current stack for next hand
        self.accumulated_reward += reward # Keep track of total reward over training
        
        # Terminal state is not explicitly constructed here for replay, as DQN uses None for next_state
        
        for i in range(len(self.history)):
            processed_state, action_taken = self.history[i] 
            next_processed_state = self.history[i+1][0] if i < len(self.history) - 1 else None
            # Reward is for the whole hand, applied to all transitions of that hand
            if self.best_response_mode: # Check if BR was active for this hand
                self.store_transition(processed_state, action_taken, reward, next_processed_state)
        
        self.update() # Call training step
        
        if self.episode % 1000 == 0 and self.episode > 0: # Save model periodically
            self.save_model()
            
        if self.hand_type_str: # Ensure it was set
            self.card_reward_stat[self.hand_type_str] = self.card_reward_stat.get(self.hand_type_str, 0) + reward

    def get_position_type(self, relative_position_scalar, num_players_in_game):
        # relative_position_scalar: 0=dealer, 1=SB, etc.
        # num_players_in_game: current number of players in the hand
        if num_players_in_game <= 3:
            if relative_position_scalar == 0: return 'late' # Button
            else: return 'blinds' 
        else: # 4+ players
            if relative_position_scalar == 0: return 'late' # Button
            elif relative_position_scalar == 1: return 'blinds' # SB
            elif relative_position_scalar == 2: return 'blinds' # BB
            # For 6max: UTG (pos 3), MP (pos 4), CO (pos 5), BTN (pos 0), SB (pos 1), BB (pos 2)
            # Dealer is pos 0. SB is pos 1. BB is pos 2. UTG for 6max is pos 3 (relative to dealer).
            # This needs to be robust for different num_players.
            # Let's use standard poker position names based on seats from button.
            # Assuming 6-max like table structure even if fewer players.
            # If num_players_in_game is small, it collapses.
            # E.g. 6-max: 0=BTN, 1=SB, 2=BB, 3=UTG, 4=MP, 5=CO
            # Relative to Dealer (pos 0): SB is 1, BB is 2.
            # UTG is btn_pos + 3 % num_players.
            # The exact naming depends on how pypokerengine defines dealer_btn and seat order.
            # Simplified:
            if relative_position_scalar <= 2 : return 'blinds' # SB, BB (or BTN if heads-up and BTN is SB)
            # Rough division for other positions
            # This is a simple heuristic. More precise would be needed for detailed positional play.
            elif relative_position_scalar < num_players_in_game / 2.0 + 1: # Early to Middle
                 return 'early' 
            else: # Middle to Late
                return 'late'

def rank_value(rank_char): # Used by initialize_card_stats
    values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
              '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return values[rank_char]
