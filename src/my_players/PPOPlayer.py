import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random as rand
from collections import deque
import math

from pypokerengine.players import BasePokerPlayer

# --- Hyperparameters (Adjust these carefully) ---
PPO_LEARNING_RATE_ACTOR_CRITIC = 1e-4
PPO_GAMMA = 0.99  # Discount factor for rewards
PPO_EPSILON_CLIP = 0.2  # Clipping parameter for PPO
PPO_K_EPOCHS = 4  # Number of epochs to update policy per iteration
PPO_BATCH_SIZE = 64 # Batch size for PPO updates
PPO_MEMORY_SIZE = 20000 # Max size of replay buffer
PPO_ENTROPY_COEF = 0.01 # Entropy bonus coefficient
PPO_GAE_LAMBDA = 0.95 # Lambda for GAE (if used, otherwise for advantage calculation)
MIN_BUFFER_SIZE_FOR_TRAINING_PPO = 1000

MAX_PLAYERS = 6 # From NFSPPlayer, ensure consistency

# --- Reusable Utility Functions (from NFSPPlayer for consistency) ---
def card_to_int(card):
    suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3}
    rank_map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}
    suit = card[0]
    rank_char = card[1]
    if rank_char == '1' and len(card) == 3 and card[2] == '0': # Handle '10'
        rank_char = 'T'
    return suit_map[suit] * 13 + rank_map[rank_char]

def community_card_to_tuple_indices(community_card_pokerlib):
    new_community_card_indices = [card_to_int(card_str) for card_str in community_card_pokerlib]
    while len(new_community_card_indices) < 5:
        new_community_card_indices.append(52) # Padding token
    return tuple(new_community_card_indices)

# --- Stat Initialization Helper Functions (from NFSPPlayer) ---
def initialize_card_stats_static(): # Renamed to avoid collision if PPOPlayer also has a method
    card_stats = {}
    ranks_str_ordered = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    for r1_char_idx, r1_char in enumerate(ranks_str_ordered):
        card_stats[r1_char + r1_char] = 0
        for r2_char_idx in range(r1_char_idx + 1, len(ranks_str_ordered)):
            r2_char = ranks_str_ordered[r2_char_idx]
            card_stats[r1_char + r2_char + 's'] = 0
            card_stats[r1_char + r2_char + 'o'] = 0
    return card_stats

def initialize_card_action_stats_static(): # Renamed
    card_action_stats = {
        'preflop': {}, 'flop': {}, 'turn': {}, 'river': {}
    }
    for street in card_action_stats.keys():
        card_action_stats[street]['raise'] = initialize_card_stats_static()
        card_action_stats[street]['call'] = initialize_card_stats_static()
        card_action_stats[street]['check'] = initialize_card_stats_static()
        card_action_stats[street]['fold'] = initialize_card_stats_static()
    return card_action_stats

def get_hand_type_str_static(hole_card_pokerlib_format): # Renamed, uses global card_to_int
    ranks_map_to_char = {0:'2', 1:'3', 2:'4', 3:'5', 4:'6', 5:'7', 6:'8', 7:'9', 8:'T', 9:'J', 10:'Q', 11:'K', 12:'A'}
    card1_idx = card_to_int(hole_card_pokerlib_format[0]) # Uses the card_to_int defined in PPOPlayer.py
    card2_idx = card_to_int(hole_card_pokerlib_format[1])
    rank1_val = card1_idx % 13
    rank2_val = card2_idx % 13
    suit1_val = card1_idx // 13
    suit2_val = card2_idx // 13
    r1_char = ranks_map_to_char[rank1_val]
    r2_char = ranks_map_to_char[rank2_val]
    if rank1_val < rank2_val:
        r1_char, r2_char = r2_char, r1_char
    if r1_char == r2_char:
        return r1_char + r2_char
    elif suit1_val == suit2_val:
        return r1_char + r2_char + 's'
    else:
        return r1_char + r2_char + 'o'

# --- PPO Replay Memory ---
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = [] # For terminal states
        self.next_states = [] # For GAE or n-step returns

    def store_experience(self, state, action, log_prob, reward, value, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        if next_state is None:
            # Assuming 'state' is a list or tuple representing features
            # Create a placeholder of zeros with the same length as a state
            self.next_states.append([0.0] * len(state)) 
        else:
            self.next_states.append(next_state)

    def sample_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, PPO_BATCH_SIZE)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+PPO_BATCH_SIZE] for i in batch_start]

        return np.array(self.states),\
               np.array(self.actions),\
               np.array(self.log_probs),\
               np.array(self.rewards),\
               np.array(self.values),\
               np.array(self.dones),\
               np.array(self.next_states),\
               batches

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.states)

# --- Actor-Critic Network ---
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_raw_feat_dim, num_actions, lstm_hidden_size=128, card_embedding_dim=16, num_card_indices=7):
        super(ActorCriticNetwork, self).__init__()
        
        # Ensure input_raw_feat_dim is an integer
        if not isinstance(input_raw_feat_dim, int):
            if isinstance(input_raw_feat_dim, str):
                try:
                    # Attempt to convert if it's a string representation of an int
                    input_raw_feat_dim_int = int(input_raw_feat_dim)
                    print(f"Warning: input_raw_feat_dim was string '{input_raw_feat_dim}', converted to int {input_raw_feat_dim_int}.")
                    input_raw_feat_dim = input_raw_feat_dim_int
                except ValueError:
                    raise TypeError(
                        f"ActorCriticNetwork: input_raw_feat_dim='{input_raw_feat_dim}' is a string and cannot be converted to int. "
                        f"This likely indicates an issue with 'num_feats' configuration or inference in PPOPlayer."
                    )
            else:
                raise TypeError(
                    f"ActorCriticNetwork: input_raw_feat_dim must be an integer. "
                    f"Got {input_raw_feat_dim} (type: {type(input_raw_feat_dim)}). "
                    f"This might be due to a failed 'num_feats' inference in PPOPlayer."
                )

        # Ensure num_card_indices is an integer
        # (The default is 7, an int. If it was passed as bool True/False, this handles it.)
        if isinstance(num_card_indices, bool):
            # If it was a boolean flag, True might mean use card_feat_dim, False means 0.
            # However, the variable name suggests a count. Let's assume True -> 1, False -> 0 if it was bool.
            # Given the error log mentioned 'bool', this handles that specific case.
            # If it's already an int (like default 7), this block is skipped.
            print(f"Warning: num_card_indices was boolean ({num_card_indices}), converting to int ({int(num_card_indices)}). This might not be the intended number of card indices.")
            self.num_card_indices = int(num_card_indices)
        elif not isinstance(num_card_indices, int):
            raise TypeError(f"ActorCriticNetwork: num_card_indices must be an integer. Got {num_card_indices} (type: {type(num_card_indices)})")
        else:
            self.num_card_indices = num_card_indices

        self.card_embedding_dim = card_embedding_dim
        self.other_features_dim = input_raw_feat_dim - self.num_card_indices

        self.card_embedding = nn.Embedding(53, self.card_embedding_dim) # 52 cards + 1 padding

        lstm_input_dim = (self.num_card_indices * self.card_embedding_dim) + self.other_features_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, batch_first=True)

        # Shared layers after LSTM
        self.fc_shared1 = nn.Linear(lstm_hidden_size, 256)
        self.ln_shared1 = nn.LayerNorm(256)
        self.fc_shared2 = nn.Linear(256, 128)
        self.ln_shared2 = nn.LayerNorm(128)

        # Actor head
        self.actor_fc = nn.Linear(128, num_actions)

        # Critic head
        self.critic_fc = nn.Linear(128, 1)

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

        lstm_out, hidden_next = self.lstm(combined_features_unsqueezed, hidden)
        lstm_out_last_step = lstm_out[:, -1, :]

        shared_out = F.relu(self.ln_shared1(self.fc_shared1(lstm_out_last_step)))
        shared_out = F.relu(self.ln_shared2(self.fc_shared2(shared_out)))

        action_probs = F.softmax(self.actor_fc(shared_out), dim=-1)
        state_value = self.critic_fc(shared_out)

        return action_probs, state_value, hidden_next

# --- PPO Player ---
class PPOPlayer(BasePokerPlayer):
    def __init__(self, shared_actor_critic_net, shared_optimizer,
                 model_save_path, optimizer_save_path, 
                 training=True,
                 initial_stack=100, # Added initial_stack
                 small_blind=1,     # Added small_blind (can be used for reference if needed)
                 card_embedding_dim=16, num_feats=85, 
                 num_card_indices_in_state=7, num_actions=11, 
                 lstm_hidden_size=128): # card_embedding_dim is a network param, but kept for consistency if player logic needs it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Store paths for the shared model/optimizer
        self.model_save_path = model_save_path
        self.optimizer_save_path = optimizer_save_path
        
        self.training = training
        
        self.initial_stack = initial_stack # Store initial_stack
        self.small_blind_ref = small_blind # Store small_blind for reference

        self.num_feats = num_feats 
        self.num_card_indices_in_state = num_card_indices_in_state 
        self.num_actions = num_actions 
        self.lstm_hidden_size = lstm_hidden_size # Used for LSTM hidden state initialization

        # Use the shared network and optimizer
        self.actor_critic_net = shared_actor_critic_net
        self.optimizer = shared_optimizer
        
        self.memory = PPOMemory()

        self.hole_card_tuple_raw_indices = None
        self.current_lstm_hidden = None
        self.episode_transitions = [] # Store (state, action, log_prob, reward, value, done, next_state) for current hand

        self.stack = self.initial_stack # Initialize stack with initial_stack
        self.raisesizes = [0.33, 0.5, 0.75, 1, 1.5, 2.0, 2.5, 3.0] # From NFSPPlayer
        self.update_count = 0
        self.last_loss_actor = None
        self.last_loss_critic = None
        self.last_loss_entropy = None
        self.loss = None # For combined loss logging
        self.accumulated_reward = 0

        # Stats for logging consistency (mimicking NFSPPlayer)
        self.hand_count = 0
        self.VPIP = 0
        self.last_vpip_action = None # Reset per hand
        self.PFR = 0
        self.last_pfr_action = None # Reset per hand
        self.three_bet = 0
        self.last_3_bet_action = None # Reset per hand
        
        self.action_stat = {
            'preflop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'flop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'turn': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'river': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0}
        }
        self.hand_type_str = None
        self.card_reward_stat = initialize_card_stats_static()
        self.card_action_stat = initialize_card_action_stats_static()
        self.street_first_action = None # To track if first action on a street has been recorded for stats

        self.load_model()

        if self.training:
            self.actor_critic_net.train()
        else:
            self.actor_critic_net.eval()

    def save_model(self):
        # Save the shared network and optimizer
        torch.save(self.actor_critic_net.state_dict(), self.model_save_path)
        torch.save(self.optimizer.state_dict(), self.optimizer_save_path)
        print(f"PPO Shared Model saved to {self.model_save_path} and optimizer to {self.optimizer_save_path}")

    def load_model(self):
        # Load the shared network and optimizer
        # Note: Loading is typically done once by the training script before distributing the shared net.
        # If called by each agent, it would just reload the same shared state.
        try:
            self.actor_critic_net.load_state_dict(torch.load(self.model_save_path, map_location=self.device, weights_only=True))
            self.optimizer.load_state_dict(torch.load(self.optimizer_save_path, map_location=self.device))
            print(f"PPO Shared Model loaded from {self.model_save_path} and optimizer from {self.optimizer_save_path}")
        except Exception as e:
            print(f"Could not load PPO Shared Model from {self.model_save_path} or optimizer from {self.optimizer_save_path}: {e}. Using fresh/current shared model state.")

    # --- State Processing & Action (Reused/Adapted from NFSP) ---
    def get_state(self, community_card_indices_tuple, round_state):
        # This should be identical to your NFSPPlayer.get_state
        # For brevity, assuming it's available and correctly constructs the raw state tuple
        # Copied a simplified version for placeholder from NFSP, ensure your full logic is here.
        current_state_list = list(self.hole_card_tuple_raw_indices + community_card_indices_tuple)
        main_pot_total = round_state['pot']['main']['amount']
        for side_pot in round_state['pot'].get('side', []): main_pot_total += side_pot['amount']
        current_state_list.append(main_pot_total)
        player_stacks_list = [0.0] * MAX_PLAYERS
        player_statuses_list = [0.0] * MAX_PLAYERS
        player_uuids = [p['uuid'] for p in round_state['seats']]
        for i in range(MAX_PLAYERS):
            if i < len(round_state['seats']):
                seat = round_state['seats'][i]
                player_stacks_list[i] = seat['stack']
                player_statuses_list[i] = 1.0 if seat['state'] == 'participating' else 0.0
        current_state_list.extend(player_stacks_list)
        current_state_list.append(round_state['seats'][self.player_id]['stack'])
        last_amounts_list = [0.0] * (MAX_PLAYERS * 4)
        # ... (Fill last_amounts_list as in NFSPPlayer) ...
        current_state_list.extend(last_amounts_list)
        current_state_list.extend(player_statuses_list)
        my_seat_idx_in_game = self.player_id
        dealer_btn_seat_idx = round_state['dealer_btn']
        num_game_players = len(round_state['seats'])
        my_position_relative_to_dealer = (my_seat_idx_in_game - dealer_btn_seat_idx + num_game_players) % num_game_players
        current_state_list.append(float(my_position_relative_to_dealer))
        # ... (SPR, Pot odds, Active players, Effective stack, Opponent street history as in NFSPPlayer) ...
        # For this example, we'll assume these are correctly added.
        # Ensure the total number of features matches self.num_feats.
        # Pad if necessary to ensure fixed length if not all features are present.
        while len(current_state_list) < self.num_feats - self.num_card_indices_in_state + 7 : # 7 is for cards
             current_state_list.append(0.0) # Pad missing features
        if len(current_state_list) > self.num_feats - self.num_card_indices_in_state + 7:
            current_state_list = current_state_list[:self.num_feats - self.num_card_indices_in_state + 7]

        return tuple(current_state_list)

    def process_state(self, s_tuple): # s_tuple is raw state from get_state
        s_list = list(s_tuple)
        
        # Dynamic normalization values based on initial_stack
        norm_pot_avg = self.initial_stack * 2.0 
        norm_pot_range = self.initial_stack * 2.0 
        norm_stack_avg = self.initial_stack
        norm_stack_range = self.initial_stack 
        norm_bet_avg = self.initial_stack / 2.0 
        norm_bet_range = self.initial_stack / 2.0
    
        # s_list[7] = Pot size
        if len(s_list) > 7: s_list[7] = (s_list[7] - norm_pot_avg) / norm_pot_range
        
        # s_list[8-13] = Player stacks (6 features)
        for i in range(8, 14):
            if len(s_list) > i: s_list[i] = (s_list[i] - norm_stack_avg) / norm_stack_range
        
        # s_list[14] = Self stack
        if len(s_list) > 14: s_list[14] = (s_list[14] - norm_stack_avg) / norm_stack_range
        
        # s_list[15-38] = last_amounts (24 features, 6 players * 4 streets)
        for i in range(15, 39):
            if len(s_list) > i: s_list[i] = (s_list[i] - norm_bet_avg) / norm_bet_range
    
        # s_list[39-44] = player_statuses_list (6 features): already 0 or 1
    
        pos_scalar_original_idx = 7 + 38 
        if pos_scalar_original_idx < len(s_list):
            pos_scalar = int(s_list[pos_scalar_original_idx])
            pos_one_hot = [0.0] * MAX_PLAYERS
            if 0 <= pos_scalar < MAX_PLAYERS:
                pos_one_hot[pos_scalar] = 1.0
            s_list = s_list[:pos_scalar_original_idx] + pos_one_hot + s_list[pos_scalar_original_idx+1:]
        
        base_idx_after_pos = pos_scalar_original_idx + MAX_PLAYERS
    
        # SPR: (spr - 10) / 10
        if base_idx_after_pos < len(s_list):
            s_list[base_idx_after_pos] = (s_list[base_idx_after_pos] - 10.0) / 10.0
        
        # Pot odds: (odds - 0.33) / 0.15
        if base_idx_after_pos + 1 < len(s_list):
            s_list[base_idx_after_pos + 1] = (s_list[base_idx_after_pos + 1] - 0.33) / 0.15
    
        # Active players count: (count - 3) / 2
        if base_idx_after_pos + 2 < len(s_list):
                s_list[base_idx_after_pos + 2] = (s_list[base_idx_after_pos + 2] - 3.0) / 2.0
        
        # Effective stack
        if base_idx_after_pos + 3 < len(s_list):
            s_list[base_idx_after_pos + 3] = (s_list[base_idx_after_pos + 3] - norm_stack_avg) / norm_stack_range
        
        street_hist_start_idx = base_idx_after_pos + 4 
    
        num_opp_street_hist_sets = 5 
        features_per_opp_street_hist = 6 
        for opp_set in range(num_opp_street_hist_sets):
            base_street_hist_idx = street_hist_start_idx + (opp_set * features_per_opp_street_hist)
            # Normalize counts (e.g. max 3 actions) for call, bet, raise
            if len(s_list) > base_street_hist_idx + 4: 
                for i in range(2, 5): # Indices for call_count, bet_count, raise_count within the 6 features
                    if len(s_list) > base_street_hist_idx +i :
                         s_list[base_street_hist_idx + i] = s_list[base_street_hist_idx + i] / 3.0
            # Normalize total_invested_this_street
            if len(s_list) > base_street_hist_idx + 5: 
                s_list[base_street_hist_idx + 5] = (s_list[base_street_hist_idx + 5] - norm_bet_avg) / norm_bet_range
        
        # Expected final size: 7(cards) + 38(pot,stacks,hist,status) + 6(pos_onehot) + 4(spr,odds,active,eff) + 30(street_hist_opp) = 85
        return tuple(s_list)

    def declare_action(self, valid_actions, hole_card_pokerlib, round_state):
        hole_card_idx1 = card_to_int(hole_card_pokerlib[0])
        hole_card_idx2 = card_to_int(hole_card_pokerlib[1])
        self.hole_card_tuple_raw_indices = (hole_card_idx1, hole_card_idx2)
        community_card_indices_tuple = community_card_to_tuple_indices(round_state['community_card'])

        raw_state_tuple = self.get_state(community_card_indices_tuple, round_state)
        # IMPORTANT: Ensure process_state produces a state compatible with your network input
        # and that its output length matches self.num_feats.
        current_processed_state = self.process_state(raw_state_tuple) 
        
        # Ensure state has correct number of features for the network. Pad if necessary.
        # This is a simplified padding, ensure it matches how you handle variable feature sets.
        state_list_for_net = list(current_processed_state)
        expected_len = self.num_feats
        if len(state_list_for_net) < expected_len:
            state_list_for_net.extend([0.0] * (expected_len - len(state_list_for_net)))
        elif len(state_list_for_net) > expected_len:
            state_list_for_net = state_list_for_net[:expected_len]
        
        state_tensor = torch.tensor([state_list_for_net], device=self.device, dtype=torch.float)

        action_probs, state_value, next_lstm_hidden = self.actor_critic_net(state_tensor, self.current_lstm_hidden)
        self.current_lstm_hidden = next_lstm_hidden # Update hidden state for next step in sequence

        # Create a mask for valid actions
        valid_action_mask = np.zeros(self.num_actions, dtype=bool)
        valid_action_mask[0] = any(va['action'] == 'fold' for va in valid_actions) # Fold
        valid_action_mask[1] = any(va['action'] == 'call' for va in valid_actions) # Call/Check
        
        # Handle raise options (Simplified from NFSPPlayer's declare_action)
        # This part needs to map your self.raisesizes to available discrete actions (action_index 2 onwards)
        # and check against valid_actions[2]['amount']['min'] and ['max']
        can_raise = any(va['action'] == 'raise' and va['amount']['min'] != -1 for va in valid_actions)
        if can_raise:
            valid_action_mask[2:] = True # Simplification: assume all raise slots are potentially valid if raise is possible

        # Apply mask to action probabilities
        masked_action_probs = action_probs.cpu().detach().numpy().flatten()
        masked_action_probs[~valid_action_mask] = 0.0
        if np.sum(masked_action_probs) == 0: # If all valid actions have 0 prob (or no valid actions)
            # Fallback: e.g., uniform over valid, or prioritize fold/call
            if valid_action_mask[0]: masked_action_probs[0] = 1.0 # Prioritize fold
            elif valid_action_mask[1]: masked_action_probs[1] = 1.0 # Prioritize call
            else: # Should not happen if fold is always an option
                 idx = np.where(valid_action_mask)[0]
                 if len(idx)>0 : masked_action_probs[idx[0]] = 1.0
                 else: masked_action_probs[0] = 1.0 # ultimate fallback
        
        # Normalize probs after masking
        prob_sum = np.sum(masked_action_probs)
        if prob_sum > 0:
            final_probs = masked_action_probs / prob_sum
        else: # If sum is still zero (e.g. no valid actions, which is unlikely)
            final_probs = np.zeros_like(masked_action_probs)
            final_probs[0] = 1.0 # Default to fold


        dist = Categorical(probs=torch.tensor(final_probs, device=self.device, dtype=torch.float))
        action_index = dist.sample()
        log_prob = dist.log_prob(action_index)

        # Store transition part for this step (reward will be added later)
        # state, action, log_prob, value, done (always False until end of hand)
        self.episode_transitions.append({
            "state": state_list_for_net, # Store the version fed to network
            "action": action_index.item(),
            "log_prob": log_prob.item(),
            "value": state_value.item(),
            "reward": 0, # Placeholder, filled at end of hand or intermediate steps
            "done": False,
            "next_state": None # Placeholder, filled at end of hand
        })
        
        # --- Map action_index to game action string and amount (like in NFSPPlayer) ---
        action_str, amount = "fold", 0
        if action_index.item() == 1: # Call/Check
            action_str = "call"
            call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
            if call_action: amount = call_action["amount"]
        elif action_index.item() > 1: # Raise
            action_str = "raise"
            raise_main_action = next((va for va in valid_actions if va['action'] == 'raise'), None)
            if raise_main_action and raise_main_action['amount']['min'] != -1 :
                # This mapping needs to be robust, similar to NFSPPlayer's logic for mapping
                # action_index (2 to N_ACTIONS-1) to specific raise amounts.
                # For simplicity, using min_raise if specific option not found.
                # You'll need to adapt the logic from NFSPPlayer that generates distinct_raise_amounts
                # and maps action_index-2 to these.
                pot_size = round_state['pot']['main']['amount'] # Simplified pot for raise calc
                min_r = raise_main_action['amount']['min']
                max_r = raise_main_action['amount']['max']
                
                raise_idx = action_index.item() - 2
                if raise_idx < len(self.raisesizes):
                    val = self.raisesizes[raise_idx] * pot_size
                    if min_r != -1 and val < min_r: val = min_r
                    if max_r != -1 and val > max_r: val = max_r
                    amount = math.floor(val)
                else: # Fallback if raise_idx out of bounds
                    amount = min_r
                
                if amount < min_r and min_r != -1: amount = min_r
                if amount > max_r and max_r != -1: amount = max_r

                if amount == 0 or (min_r != -1 and amount < min_r) : # If computed amount is invalid, try to call or fold
                    if valid_action_mask[1]: # If call is valid
                        action_str = "call"
                        call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
                        if call_action: amount = call_action["amount"]
                    else: # Fold
                        action_str = "fold"
                        amount = 0
            else: # Cannot raise (e.g. min_raise is -1), try to call or fold
                if valid_action_mask[1]:
                    action_str = "call"
                    call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
                    if call_action: amount = call_action["amount"]
                else:
                    action_str = "fold"
                    amount = 0
        
        # VPIP, PFR, 3-Bet tracking (similar to NFSPPlayer)
        # This logic is simplified from NFSP. PPOPlayer already has VPIP/PFR/3bet attributes.
        # The hand_count, VPIP, PFR, three_bet attributes are incremented here.
        # Ensure these are reset by training.py as expected. training.py resets them.

        current_street = round_state["street"]
        stats_action_to_record = 'check' if action_str == "call" and amount == 0 else action_str

        # Simplified stat updates, similar to parts of NFSP.
        # NFSPPlayer updates action_stat and card_action_stat only for the *first* action on a street.
        if not self.street_first_action:
            self.action_stat[current_street][stats_action_to_record] += 1
            if self.hand_type_str:
                 # Ensure hand_type_str exists as a key, useful if new hand types could appear
                if self.hand_type_str not in self.card_action_stat[current_street][stats_action_to_record]:
                    self.card_action_stat[current_street][stats_action_to_record][self.hand_type_str] = 0
                self.card_action_stat[current_street][stats_action_to_record][self.hand_type_str] += 1
            self.street_first_action = stats_action_to_record
        
        # VPIP, PFR, 3-Bet (simplified, assuming training.py resets VPIP, PFR, three_bet, hand_count)
        # hand_count is incremented once per hand in receive_round_start now.
        if current_street == 'preflop':
            # VPIP: any voluntary money in pot (call or raise)
            if action_str in ["call", "raise"] and amount > 0:
                if self.last_vpip_action is None: # Count only first voluntary action for VPIP this hand
                    self.VPIP += 1
                    self.last_vpip_action = action_str 

            if action_str == "raise":
                # PFR: first raise preflop
                if self.last_pfr_action is None: # Count only first PFR action this hand
                    self.PFR += 1
                    self.last_pfr_action = action_str
                
                # 3-Bet: a reraise over an initial raise by another player
                if self.last_3_bet_action is None: # Count only first 3-bet action this hand
                    # Check if there was at least one raise by ANOTHER player before our current raise action.
                    # round_state['action_histories']['preflop'] contains actions *before* this one.
                    pre_flop_actions_history = round_state['action_histories'].get('preflop', [])
                    is_already_raised_by_opponent = any(
                        act['action'] == 'RAISE' and act['uuid'] != self.uuid for act in pre_flop_actions_history
                    )
                    # Ensure this is our first raise as well to be a 3-bet (not a 4-bet initiated by us)
                    is_our_first_raise_this_hand_preflop = not any(
                        act['action'] == 'RAISE' and act['uuid'] == self.uuid for act in pre_flop_actions_history
                    )

                    if is_already_raised_by_opponent and is_our_first_raise_this_hand_preflop:
                        self.three_bet += 1
                        self.last_3_bet_action = action_str
        
        return action_str, math.floor(amount)

    def update_ppo(self):
        if len(self.memory) < MIN_BUFFER_SIZE_FOR_TRAINING_PPO :
            return

        states, actions, old_log_probs, rewards, old_values, dones, next_states, batches = self.memory.sample_batch()
        
        # Calculate advantages using GAE or simple reward-to-go - V(s)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        future_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # For terminal states, next_value is 0. Reward is the terminal reward.
                delta = rewards[t] - old_values[t] # No V(s_next)
                future_advantage = 0 # Reset GAE for new episode
            else:
                # Get V(s_next). If next_states[t] is None (should not happen if dones[t] is False and handled properly),
                # then V(s_next) might be estimated as 0 or from a subsequent state if available.
                # Here, we assume old_values for next step can be estimated or is the next value in trajectory.
                # A more robust way: re-evaluate V(s_next) if needed, or ensure old_values[t+1] is V(s_next)
                # For simplicity with current PPOMemory, we use next step's old_value or 0 if terminal.
                next_val = old_values[t+1] if t + 1 < len(old_values) and not dones[t+1] else 0.0
                delta = rewards[t] + PPO_GAMMA * next_val * (1 - dones[t]) - old_values[t] # dones[t] for current step
            
            future_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * future_advantage * (1-dones[t]) # dones[t] for current step
            advantages[t] = future_advantage
        
        advantages_tensor = torch.tensor(advantages, dtype=torch.float, device=self.device)
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        states_tensor = torch.tensor(states, dtype=torch.float, device=self.device)
        # old_values_tensor = torch.tensor(old_values, dtype=torch.float, device=self.device) # Not directly used in loss like this
        
        # The rewards used for value target should be discounted returns
        returns = advantages + old_values # R_to_go = Advantage + V(s)
        returns_tensor = torch.tensor(returns, dtype=torch.float, device=self.device)


        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0

        for _ in range(PPO_K_EPOCHS):
            for batch_indices in batches:
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices] # Target for value function

                # Forward pass to get current policy probs and values
                # No hidden state passed for batch training; LSTM statefulness is per-trajectory
                action_probs, current_values, _ = self.actor_critic_net(batch_states, hidden=None)
                current_values = current_values.squeeze(1)

                dist = Categorical(probs=action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Actor Loss (PPO Clipped Objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_EPSILON_CLIP, 1 + PPO_EPSILON_CLIP) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Loss (MSE)
                critic_loss = F.mse_loss(current_values, batch_returns)

                # Total Loss
                loss = actor_loss + 0.5 * critic_loss - PPO_ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic_net.parameters(), 0.5) # Grad clipping
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()
        
        num_updates = PPO_K_EPOCHS * len(batches)
        self.last_loss_actor = total_actor_loss / num_updates
        self.last_loss_critic = total_critic_loss / num_updates
        self.last_loss_entropy = total_entropy_loss / num_updates
        self.loss = self.last_loss_actor + self.last_loss_critic # Set combined loss

        self.memory.clear_memory()
        self.update_count +=1

    # --- PyPokerEngine Interface Methods ---
    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']
        for i in range(len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i; break
        self.stack = game_info['seats'][self.player_id]['stack']

    def receive_round_start_message(self, round_count, hole_card_pokerlib, seats):
        self.episode_transitions = [] # Clear transitions for the new hand
        # Reset LSTM hidden state
        self.current_lstm_hidden = (torch.zeros(1, 1, self.lstm_hidden_size).to(self.device),
                                    torch.zeros(1, 1, self.lstm_hidden_size).to(self.device))
        # Reset stats
        self.last_vpip_action = None
        self.last_pfr_action = None
        self.last_3_bet_action = None
        self.street_first_action = None # Reset for the new hand/round
        self.hand_type_str = get_hand_type_str_static(hole_card_pokerlib) # Use static helper
        self.hand_count +=1 # Increment hand_count here, once per hand.

    def receive_street_start_message(self, street, round_state):
        self.street_first_action = None # Reset for new street

    def receive_game_update_message(self, new_action, round_state):
        pass # Action is declared by declare_action, experience stored there

    def receive_round_result_message(self, winners, hand_info, round_state):
        if not self.training or not self.episode_transitions:
            return

        final_stack = 0
        for player_seat_info in round_state['seats']:
            if player_seat_info['uuid'] == self.uuid:
                final_stack = player_seat_info['stack']
                break
        
        # Calculate reward for the entire hand
        reward_for_hand = (final_stack - self.stack) / self.initial_stack # Normalize by initial_stack
        self.stack = final_stack # Update stack for next hand calculation
        self.accumulated_reward += reward_for_hand

        # Update card_reward_stat
        if self.hand_type_str:
            # Ensure hand_type_str exists as a key
            if self.hand_type_str not in self.card_reward_stat:
                self.card_reward_stat[self.hand_type_str] = 0
            self.card_reward_stat[self.hand_type_str] += reward_for_hand
            
        # Assign reward to the last transition and mark as done
        if self.episode_transitions:
            # Distribute reward: PPO typically uses reward-to-go.
            # For simplicity here, assign final reward to all steps, or just last step.
            # A better approach is to calculate discounted rewards for each step.
            # The GAE calculation in update_ppo already handles this.
            
            # Mark the last transition as done and set its reward
            self.episode_transitions[-1]["reward"] = reward_for_hand
            self.episode_transitions[-1]["done"] = True
            
            # Populate next_state for all but the last transition
            for i in range(len(self.episode_transitions) - 1):
                self.episode_transitions[i]["next_state"] = self.episode_transitions[i+1]["state"]
                # Intermediate rewards could be 0, or a small shaping reward if designed.
                # For now, only final reward is used for the whole trajectory by GAE.

            # Store all transitions from the completed hand into the main PPO memory
            for trans in self.episode_transitions:
                self.memory.store_experience(
                    trans["state"], trans["action"], trans["log_prob"],
                    trans["reward"], trans["value"], trans["done"], trans["next_state"]
                )
        
        self.episode_transitions = [] # Clear for next hand

        if len(self.memory) >= PPO_BATCH_SIZE : # Or some other condition, e.g. PPO_MEMORY_SIZE reached
            self.update_ppo()

        if self.update_count % 100 == 0 and self.update_count > 0 : # Save model periodically
             print(f"PPO Update Count: {self.update_count}, Actor Loss: {self.last_loss_actor:.4f}, Critic Loss: {self.last_loss_critic:.4f}")
             self.save_model()

    # --- Dummy methods if required by BasePokerPlayer ---
    def declare_fold(self, valid_actions, hole_card, round_state): # Not used if declare_action handles all
        return "fold", 0
    def declare_call(self, valid_actions, hole_card, round_state): # Not used
        return "call", valid_actions[1]['amount']
    def declare_raise(self, valid_actions, hole_card, round_state): # Not used
        return "raise", valid_actions[2]['amount']['min']
