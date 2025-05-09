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
anticipatory_param = 0.1  # Probability of using best response policy
eta = 0.1  # Learning rate for average policy updates
target_net_update_freq = 10000
epsilon_start = 0.1
epsilon_end = 0.01
epsilon_decay = 500000
min_buffer_size_for_training = 1000

# Position constants to make code more readable
EARLY_POSITION = 0
MIDDLE_POSITION = 1
LATE_POSITION = 2
BLINDS = 3

class OpponentTracker:
    """Tracks opponent tendencies and statistics across multiple hands"""
    def __init__(self, num_players):
        self.num_players = num_players
        self.player_stats = {}
        self.reset_stats()
        
    def reset_stats(self):
        """Initialize or reset opponent statistics"""
        self.player_stats = {
            i: {
                'vpip': 0,           # Voluntarily Put $ In Pot
                'vpip_opp': 0,       # Opportunities to VPIP
                'pfr': 0,            # Pre-Flop Raise
                'pfr_opp': 0,        # Opportunities to PFR
                'three_bet': 0,      # 3-Bet frequency
                'three_bet_opp': 0,  # Opportunities to 3-Bet
                'fold_to_3bet': 0,   # Fold to 3-Bet
                'fold_to_3bet_opp': 0,# Opportunities to fold to 3-Bet
                'aggression_factor': 0, # (Raises + Bets) / Calls
                'raise_count': 0,    # Number of raises
                'bet_count': 0,      # Number of bets
                'call_count': 0,     # Number of calls
                'check_count': 0,    # Number of checks
                'fold_count': 0,     # Number of folds
                'actions_by_position': {  # Actions broken down by position
                    'early': {'fold': 0, 'call': 0, 'raise': 0, 'check': 0, 'total': 0},
                    'middle': {'fold': 0, 'call': 0, 'raise': 0, 'check': 0, 'total': 0},
                    'late': {'fold': 0, 'call': 0, 'raise': 0, 'check': 0, 'total': 0},
                    'blinds': {'fold': 0, 'call': 0, 'raise': 0, 'check': 0, 'total': 0}
                },
                'cbets': 0,          # Continuation bets
                'cbet_opps': 0,      # Continuation bet opportunities
                'hand_types_shown': defaultdict(int)  # Hand types shown at showdown
            } for i in range(self.num_players)
        }

    def update_from_game_state(self, round_state, player_id, street):
        """Update opponent statistics based on current game state"""
        actions = round_state['action_histories'].get(street, [])
        
        # Get positions for all players
        positions = self._get_positions(round_state)
        
        for action in actions:
            # Skip our own actions
            if action['uuid'] == round_state['seats'][player_id]['uuid']:
                continue
                
            # Find opponent index
            for i, seat in enumerate(round_state['seats']):
                if seat['uuid'] == action['uuid']:
                    opp_id = i
                    break
            else:
                continue  # Player not found
                
            # Get position type (early, middle, late, blinds)
            pos_type = self._position_to_type(positions[opp_id], len(round_state['seats']))
            
            # Update action counts
            action_type = action['action']
            if action_type == 'call':
                if action.get('amount', 0) == 0:
                    action_type = 'check'  # Reclassify as check if amount is 0
                    self.player_stats[opp_id]['check_count'] += 1
                else:
                    self.player_stats[opp_id]['call_count'] += 1
            elif action_type == 'raise':
                self.player_stats[opp_id]['raise_count'] += 1
            elif action_type == 'fold':
                self.player_stats[opp_id]['fold_count'] += 1
                
            # Update position-based stats
            if action_type in ['fold', 'call', 'raise', 'check']:
                self.player_stats[opp_id]['actions_by_position'][pos_type][action_type] += 1
                self.player_stats[opp_id]['actions_by_position'][pos_type]['total'] += 1

            # Update pre-flop specific stats
            if street == 'preflop':
                # VPIP
                if action_type in ['call', 'raise'] and action.get('amount', 0) > 0:
                    self.player_stats[opp_id]['vpip'] += 1
                self.player_stats[opp_id]['vpip_opp'] += 1
                
                # PFR
                if action_type == 'raise':
                    self.player_stats[opp_id]['pfr'] += 1
                self.player_stats[opp_id]['pfr_opp'] += 1
                
                # 3-Bet
                if action_type == 'raise':
                    # Check if there was a previous raise in this round
                    prev_raise = False
                    for prev_action in actions:
                        if prev_action['action'] == 'raise' and prev_action != action:
                            prev_raise = True
                            break
                            
                    if prev_raise:
                        self.player_stats[opp_id]['three_bet'] += 1
                    self.player_stats[opp_id]['three_bet_opp'] += 1
                    
                # Fold to 3-Bet
                if action_type == 'fold':
                    # Check if last action was a 3-bet
                    if len(actions) >= 2:
                        prev_action = actions[-2]
                        if prev_action['action'] == 'raise':
                            # Check if there was a raise before that
                            prev_raise = False
                            for pa in actions[:-2]:
                                if pa['action'] == 'raise':
                                    prev_raise = True
                                    break
                                    
                            if prev_raise:  # It was indeed a 3-bet
                                self.player_stats[opp_id]['fold_to_3bet'] += 1
                            self.player_stats[opp_id]['fold_to_3bet_opp'] += 1
            
            # Update continuation bet stats
            if street == 'flop':
                preflop_raiser = None
                if 'preflop' in round_state['action_histories']:
                    for action in round_state['action_histories']['preflop']:
                        if action['action'] == 'raise':
                            preflop_raiser = action['uuid']
                            
                if preflop_raiser == action['uuid'] and action_type in ['raise', 'call'] and action.get('amount', 0) > 0:
                    self.player_stats[opp_id]['cbets'] += 1
                    
                if preflop_raiser == action['uuid']:
                    self.player_stats[opp_id]['cbet_opps'] += 1
    
    def get_opponent_feature_vector(self, opp_id):
        """Create a feature vector for an opponent that can be used in decision making"""
        stats = self.player_stats[opp_id]
        
        # Calculate derived stats
        vpip_pct = stats['vpip'] / max(1, stats['vpip_opp'])
        pfr_pct = stats['pfr'] / max(1, stats['pfr_opp'])
        three_bet_pct = stats['three_bet'] / max(1, stats['three_bet_opp'])
        fold_to_3bet_pct = stats['fold_to_3bet'] / max(1, stats['fold_to_3bet_opp'])
        cbet_pct = stats['cbets'] / max(1, stats['cbet_opps'])
        
        # Calculate aggression factor
        total_aggressive = stats['raise_count'] + stats['bet_count']
        aggression_factor = total_aggressive / max(1, stats['call_count'])
        
        # Calculate positional play tendencies
        positional_raises = {
            'early': stats['actions_by_position']['early']['raise'] / max(1, stats['actions_by_position']['early']['total']),
            'middle': stats['actions_by_position']['middle']['raise'] / max(1, stats['actions_by_position']['middle']['total']),
            'late': stats['actions_by_position']['late']['raise'] / max(1, stats['actions_by_position']['late']['total']),
            'blinds': stats['actions_by_position']['blinds']['raise'] / max(1, stats['actions_by_position']['blinds']['total'])
        }
        
        # Build feature vector
        features = [
            vpip_pct,
            pfr_pct,
            three_bet_pct,
            fold_to_3bet_pct,
            aggression_factor,
            cbet_pct,
            positional_raises['early'],
            positional_raises['middle'],
            positional_raises['late'],
            positional_raises['blinds']
        ]
        
        return features
    
    def _get_positions(self, round_state):
        """Determine relative positions for all players based on dealer button"""
        n_players = len(round_state['seats'])
        dealer_pos = round_state['dealer_btn']
        
        # Create a mapping of seat index to position relative to dealer
        # 0 is button (dealer), 1 is small blind, 2 is big blind, etc.
        positions = [(i - dealer_pos) % n_players for i in range(n_players)]
        return positions
    
    def _position_to_type(self, position, n_players):
        """Convert numerical position to position type"""
        if n_players <= 3:
            if position == 0:  # Button
                return 'late'
            else:  # Blinds
                return 'blinds'
        else:
            if position == 0:  # Button
                return 'late'
            elif position <= 2:  # SB, BB
                return 'blinds'
            elif position < n_players / 3 + 2:  # First third of table after blinds
                return 'early'
            elif position < 2 * n_players / 3 + 2:  # Second third
                return 'middle'
            else:  # Last third
                return 'late'
                
    def classify_opponent(self, opp_id):
        """Classify opponent playing style"""
        stats = self.player_stats[opp_id]
        
        # Need minimum sample size
        if stats['vpip_opp'] < 10:
            return "Unknown"
            
        vpip = stats['vpip'] / max(1, stats['vpip_opp'])
        pfr = stats['pfr'] / max(1, stats['pfr_opp'])
        
        # Classic poker player classifications
        if vpip < 0.15:
            if pfr < 0.1:
                return "Rock"  # Very tight, passive
            else:
                return "TAG"  # Tight aggressive
        elif vpip < 0.30:
            if pfr < 0.2:
                return "Calling Station"  # Loose passive
            else:
                return "Solid Reg"  # Regular player
        else:  # vpip >= 0.30
            if pfr < 0.25:
                return "Fish"  # Very loose passive
            else:
                return "LAG"  # Loose aggressive
                
    def get_exploit_strategy(self, opp_id, our_position, opp_position):
        """
        Generate strategic adjustments to exploit observed opponent tendencies
        Returns a dict of strategy adjustments
        """
        style = self.classify_opponent(opp_id)
        stats = self.player_stats[opp_id]
        
        # Default adjustments (neutral)
        adjustments = {
            'value_bet_sizing': 1.0,  # Multiplier for value bets
            'bluff_frequency': 1.0,   # Multiplier for bluff frequency
            'call_threshold': 1.0,    # Multiplier for calling threshold
            'raise_threshold': 1.0,   # Multiplier for raising threshold
            'position_importance': 1.0 # How much to adjust for position
        }
        
        # Adjust based on player type
        if style == "Rock":
            # Rocks fold too much, bluff more and value bet less
            adjustments['bluff_frequency'] = 1.5
            adjustments['value_bet_sizing'] = 0.8
            adjustments['call_threshold'] = 0.9  # Call less, they rarely bluff
            
        elif style == "TAG":
            # TAGs are balanced but value-heavy, slightly more bluffs
            adjustments['bluff_frequency'] = 1.2
            adjustments['call_threshold'] = 1.1  # Call a bit more, they can bluff
            
        elif style == "Calling Station":
            # Calling stations call too much, value bet more and bluff less
            adjustments['value_bet_sizing'] = 1.3
            adjustments['bluff_frequency'] = 0.5
            
        elif style == "Fish":
            # Fish play too many hands weakly, value bet big and don't bluff
            adjustments['value_bet_sizing'] = 1.5
            adjustments['bluff_frequency'] = 0.2
            
        elif style == "LAG":
            # LAGs are aggressive, call them more and value bet more
            adjustments['call_threshold'] = 1.3
            adjustments['value_bet_sizing'] = 1.2
            adjustments['raise_threshold'] = 1.1  # Reraise them more
        
        # Position-based adjustments
        # If we're in late position vs early position, we can be more aggressive
        if our_position == LATE_POSITION and opp_position == EARLY_POSITION:
            adjustments['bluff_frequency'] *= 1.2
            adjustments['call_threshold'] *= 1.1
        # If we're in early position vs late position, we need to be tighter
        elif our_position == EARLY_POSITION and opp_position == LATE_POSITION:
            adjustments['bluff_frequency'] *= 0.8
            adjustments['call_threshold'] *= 0.9
            
        return adjustments

class ReservoirBuffer:
    """Reservoir buffer for supervised learning, maintains a uniform random sample over a data stream"""
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.size = 0
        
    def push(self, transition):
        """Add a new transition to the buffer with reservoir sampling"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            # Reservoir sampling
            idx = rand.randint(0, self.size)
            if idx < self.capacity:
                self.buffer[idx] = transition
        self.size += 1
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return rand.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class ReplayMemory:
    """Experience replay buffer for DQN"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, transition):
        """Add a new transition to the buffer"""
        self.memory.append(transition)
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return rand.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

# Best Response Network (Q-Network)
class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Q-network for best response
        self.fc1 = nn.Linear(self.input_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Average Policy Network (Supervised Learning)
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PolicyNetwork, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Policy network for average strategy
        self.fc1 = nn.Linear(self.input_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

class NFSPPlayer(BasePokerPlayer):
    def __init__(self, q_model_path, policy_model_path, q_optimizer_path, policy_optimizer_path, training=True):
        """
        Neural Fictitious Self Play (NFSP) agent for poker
        
        Args:
            q_model_path: Path to save/load the Q-network
            policy_model_path: Path to save/load the policy network
            q_optimizer_path: Path to save/load the Q-network optimizer
            policy_optimizer_path: Path to save/load the policy network optimizer
            training: Whether the agent is in training mode
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nb_player = None  # Number of players in the game
        self.player_id = None  # Player ID (ID = 0 if p1)
        self.episode = 0  # Episode counter
        
        # Initialize replay buffers
        self.br_memory = ReplayMemory(br_memory_size)  # Best response (RL) memory
        self.sl_memory = ReservoirBuffer(sl_memory_size)  # Supervised learning memory
        
        # Loss tracking
        self.q_loss = None
        self.policy_loss = None
        
        # Hyperparameters
        self.gamma = gamma
        self.anticipatory_param = anticipatory_param
        self.eta = eta
        self.batch_size = batch_size
        self.target_net_update_freq = target_net_update_freq
        
        # Epsilon greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Training-related attributes
        self.stack = 100
        self.hole_card = None
        self.q_model_path = q_model_path
        self.policy_model_path = policy_model_path
        self.q_optimizer_path = q_optimizer_path
        self.policy_optimizer_path = policy_optimizer_path
        self.update_count = 0
        self.training = training
        self.best_response_mode = None  # Will be set at the start of each hand
        self.current_state = None
        self.history = []
        
        # Player's stats (similar to DQNPlayer)
        self.hand_count = 0
        self.VPIP = 0
        self.last_vpip_action = None
        self.PFR = 0
        self.last_pfr_action = None
        self.three_bet = 0
        self.last_3_bet_action = None
        self.street_first_action = None
        
        # Adaptive raise sizes based on professional play
        self.raisesizes = [0.33, 0.5, 0.75, 1, 1.5, 2.0, 2.5, 3.0]
        
        # Track individual hand results for analysis
        self.last_reward = 0
        self.accumulated_reward = 0
        self.state = []
        
        # Initialize opponent modeling system
        self.opponent_tracker = None  # Will be initialized when we know number of players
        
        # Professional play additions
        self.hand_strength_cache = {}  # Cache for hand strength calculations
        self.position_adjustments = {
            'early': {'value_range': 0.7, 'bluff_range': 0.4},    # Tightest in early position
            'middle': {'value_range': 0.8, 'bluff_range': 0.6},   # Moderate in middle position
            'late': {'value_range': 0.9, 'bluff_range': 0.8},     # Widest in late position
            'blinds': {'value_range': 0.75, 'bluff_range': 0.5}   # Defensive in blinds
        }
        
        # Stack-to-pot ratio considerations
        self.spr_thresholds = {
            'commitment': 3.0,    # Below this SPR, committed to pot
            'postflop_play': 8.0  # Below this, be careful with drawing hands
        }
        
        # Statistics tracking 
        self.action_stat = {
            'preflop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'flop': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'turn': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0},
            'river': {'check': 0, 'call': 0, 'raise': 0, 'fold': 0}
        }
        
        self.hand_type = None
        # Initialize card_reward_stat and card_action_stat
        self.card_reward_stat = self.initialize_card_stats()
        self.card_action_stat = self.initialize_card_action_stats()
        
        # Initialize neural networks
        self.num_actions = 11  # Same as before but with more strategic options
        
        # Extend the state representation to include opponent modeling features
        self.opponent_feature_size = 10  # Size of opponent feature vector
        self.num_feats = (46 + self.opponent_feature_size * 5,)  # Extended state size
        self.initialize_networks()
        
    def initialize_card_stats(self):
        """Initialize card reward statistics dictionary"""
        # Create the same structure as in DQNPlayer
        card_stats = {}
        
        # Pocket pairs
        for rank in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']:
            card_stats[rank + rank] = 0
            
        # Suited hands
        for i, rank1 in enumerate(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3']):
            for rank2 in ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']:
                if rank_value(rank1) > rank_value(rank2):
                    card_stats[rank1 + rank2 + 's'] = 0
                    
        # Offsuit hands
        for i, rank1 in enumerate(['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3']):
            for rank2 in ['K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']:
                if rank_value(rank1) > rank_value(rank2):
                    card_stats[rank1 + rank2 + 'o'] = 0
                    
        return card_stats
    
    def initialize_card_action_stats(self):
        """Initialize card action statistics dictionary"""
        card_action_stats = {
            'preflop': {},
            'flop': {},
            'turn': {},
            'river': {}
        }
        
        for street in card_action_stats.keys():
            card_action_stats[street]['raise'] = self.initialize_card_stats()
            card_action_stats[street]['call'] = self.initialize_card_stats()
            card_action_stats[street]['check'] = self.initialize_card_stats()
            card_action_stats[street]['fold'] = self.initialize_card_stats()
            
        return card_action_stats
    
    def initialize_networks(self):
        """Initialize Q-network and policy network"""
        # Q-network and target network for best response
        self.q_net = QNetwork(self.num_feats, self.num_actions).to(self.device)
        self.target_q_net = QNetwork(self.num_feats, self.num_actions).to(self.device)
        
        # Policy network for average strategy
        self.policy_net = PolicyNetwork(self.num_feats, self.num_actions).to(self.device)
        
        # Try to load existing models
        try:
            self.q_net.load_state_dict(torch.load(self.q_model_path))
            self.policy_net.load_state_dict(torch.load(self.policy_model_path))
        except Exception as e:
            print(f"Could not load models: {e}")
            
        # Initialize target network
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate_br)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate_avg)
        
        # Try to load optimizer states
        try:
            self.q_optimizer.load_state_dict(torch.load(self.q_optimizer_path))
            self.policy_optimizer.load_state_dict(torch.load(self.policy_optimizer_path))
        except Exception as e:
            print(f"Could not load optimizer states: {e}")
            
        # Set training or evaluation mode
        if self.training:
            self.q_net.train()
            self.policy_net.train()
            self.target_q_net.train()
        else:
            self.q_net.eval()
            self.policy_net.eval()
            self.target_q_net.eval()
    
    def save_model(self):
        """Save the model and optimizer states"""
        torch.save(self.q_net.state_dict(), self.q_model_path)
        torch.save(self.policy_net.state_dict(), self.policy_model_path)
        torch.save(self.q_optimizer.state_dict(), self.q_optimizer_path)
        torch.save(self.policy_optimizer.state_dict(), self.policy_optimizer_path)
        
    def update_target_network(self):
        """Update the target network periodically"""
        if self.update_count % self.target_net_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
    def update_epsilon(self):
        """Update epsilon for epsilon-greedy exploration"""
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                       math.exp(-1. * self.update_count / self.epsilon_decay)
        
    def select_action(self, state, valid_actions):
        """
        Select an action using either best response or average policy
        
        In NFSP, with probability 'anticipatory_param', the agent uses its best response (Q-network)
        Otherwise, it uses its average policy (Policy network)
        """
        if self.best_response_mode is None:
            # At the beginning of each hand, decide whether to use BR or AVG policy
            self.best_response_mode = np.random.random() < self.anticipatory_param
            
        with torch.no_grad():
            state_tensor = torch.tensor([state], device=self.device, dtype=torch.float)
            
            # Create valid action mask
            valid_action_mask = np.ones(self.num_actions, dtype=bool)
            
            # Adjust mask based on game state
            if valid_actions[2]['amount']['max'] == -1:  # Can't raise
                valid_action_mask[2:] = False  # Only fold and call are valid
            if valid_actions[1]['amount'] == 0:  # Check is available
                valid_action_mask[0] = False  # Cannot fold
                
            if self.best_response_mode:
                # Use best response (Q-network) with epsilon-greedy
                q_values = self.q_net(state_tensor).cpu().numpy().reshape(-1)
                masked_q_values = np.where(valid_action_mask, q_values, -np.inf)
                
                if np.random.random() < self.epsilon and self.training:
                    # Random action (from valid actions)
                    valid_indices = np.where(valid_action_mask)[0]
                    action = np.random.choice(valid_indices)
                else:
                    # Greedy action
                    action = np.argmax(masked_q_values)
            else:
                # Use average policy (Policy network)
                policy_probs = self.policy_net(state_tensor).cpu().numpy().reshape(-1)
                masked_policy_probs = np.where(valid_action_mask, policy_probs, 0)
                
                # Normalize probabilities if needed
                if np.sum(masked_policy_probs) > 0:
                    masked_policy_probs = masked_policy_probs / np.sum(masked_policy_probs)
                else:
                    # Fallback: uniform distribution over valid actions
                    valid_indices = np.where(valid_action_mask)[0]
                    masked_policy_probs = np.zeros_like(policy_probs)
                    masked_policy_probs[valid_indices] = 1.0 / len(valid_indices)
                
                # Sample from policy
                action = np.random.choice(self.num_actions, p=masked_policy_probs)
                
            # Store action for supervised learning (only if best response mode)
            if self.best_response_mode and self.training:
                self.sl_memory.push((state, action))
                
            return action
    
    def store_transition(self, state, action, reward, next_state):
        """Store transition for reinforcement learning (best response)"""
        if self.training and self.best_response_mode:
            self.br_memory.push((state, action, reward, next_state))
    
    def train_q_network(self):
        """Train the Q-network from experience replay"""
        if len(self.br_memory) < min_buffer_size_for_training:
            return
            
        # Sample minibatch
        transitions = self.br_memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        
        # Convert to tensors
        state_batch = torch.tensor(batch_state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch_action, device=self.device, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch_reward, device=self.device, dtype=torch.float).unsqueeze(1)
        
        # Filter out terminal states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), 
                                    device=self.device, dtype=torch.bool)
        
        non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None],
                                           device=self.device, dtype=torch.float)
        
        # Compute Q(s_t, a)
        q_values = self.q_net(state_batch).gather(1, action_batch)
        
        # Compute max Q(s_{t+1}, a) for non-terminal states
        next_q_values = torch.zeros(self.batch_size, 1, device=self.device)
        if len(non_final_next_states) > 0:
            next_q_values[non_final_mask] = self.target_q_net(non_final_next_states).max(1, keepdim=True)[0].detach()
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        self.q_loss = loss.item()
        
        # Optimize
        self.q_optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        for param in self.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_optimizer.step()
    
    def train_policy_network(self):
        """Train the policy network from reservoir buffer (supervised learning)"""
        if len(self.sl_memory) < min_buffer_size_for_training:
            return
            
        # Sample minibatch
        transitions = self.sl_memory.sample(self.batch_size)
        batch_state, batch_action = zip(*transitions)
        
        # Convert to tensors
        state_batch = torch.tensor(batch_state, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(batch_action, device=self.device, dtype=torch.long)
        
        # Forward pass
        log_probs = torch.log(self.policy_net(state_batch) + 1e-8)  # Add small constant for numerical stability
        
        # Compute loss (negative log likelihood)
        loss = F.nll_loss(log_probs, action_batch)
        self.policy_loss = loss.item()
        
        # Optimize
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
    
    def update(self, episode=None):
        """Update both networks"""
        if not self.training:
            return
            
        self.train_q_network()
        self.train_policy_network()
        self.update_target_network()
        self.update_epsilon()
        self.update_count += 1
    
    @staticmethod
    def card_to_int(card):
        """Convert card to int, card[0]:suit, card[1]:rank"""
        suit_map = {'H': 0, 'S': 1, 'D': 2, 'C': 3}
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                    'A': 12}
        return suit_map[card[0]] * 13 + rank_map[card[1]]
    
    def get_hand_type(self, hole_card):
        """Determine the hand type (e.g., AA, AKs, AKo)"""
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
        """Convert community card to tuple of integers"""
        new_community_card = []
        for i in range(0, len(community_card)):
            new_community_card.append(self.card_to_int(community_card[i]))
        for i in range(0, 5 - len(community_card)):
            # if community card num <5, append 52 to fill out the rest
            new_community_card.append(52)
        return tuple(new_community_card)
    
    @staticmethod
    def process_state(s):
        """Normalize the state values including opponent modeling features"""
        new_s = list(s)
        
        # Normalize card values
        for i in range(0, 7):
            new_s[i] = new_s[i] / 26.0 - 1
            
        # Normalize stack and pot values
        for i in range(7, 39):
            new_s[i] = (new_s[i] - 100) / 100
            
        # Normalize position class (0-3) to range [0,1]
        if len(new_s) > 39:
            new_s[39] = new_s[39] / 3.0
            
        # Normalize SPR (Stack-to-Pot Ratio)
        if len(new_s) > 40:
            # Cap SPR at 20 for normalization
            new_s[40] = min(new_s[40], 20.0) / 20.0
            
        # Normalize pot odds
        if len(new_s) > 41:
            new_s[41] = new_s[41]  # Already in [0,1]
            
        # Normalize opponent features if present
        if len(new_s) > 42:
            # VPIP, PFR, 3bet%, etc. are already in [0,1]
            # Aggression factor needs normalization
            for i in range(46, len(new_s), 13):  # Aggression factor is at position 4 in each 13-value opponent block
                if i < len(new_s):
                    new_s[i] = min(new_s[i], 10.0) / 10.0  # Cap aggression at 10
                    
        return tuple(new_s)
    
    def get_state(self, community_card, round_state):
        """Construct the current state vector with enhanced professional features"""
        self.state = self.hole_card + community_card 

        # Basic pot and stack information
        main_pot = round_state['pot']['main']['amount']
        self.state += (main_pot,)

        player_stacks = tuple(seat['stack'] for seat in round_state['seats'])
        self.state += player_stacks  

        my_stack = round_state['seats'][self.player_id]['stack']
        self.state += (my_stack, )

        # Historical action tracking
        last_amounts = {seat['uuid']: [0, 0, 0, 0] for seat in round_state['seats']}  # 4 rounds per player

        for i, street in enumerate(['preflop', 'flop', 'turn', 'river']):
            actions = round_state['action_histories'].get(street, [])
            for action in actions:
                uuid = action['uuid']
                last_amounts[uuid][i] = action.get('amount', 0)

        for player_id in last_amounts:
            self.state += tuple(last_amounts[player_id])

        # Player participation status
        player_statuses = tuple(1 if seat['state'] == 'participating' else 0 for seat in round_state['seats'])
        self.state += player_statuses

        # Calculate relative position - a critical factor in professional play
        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                my_position = int(player['name'][-1])
                break

        dealer_position = round_state['dealer_btn']
        num_players = len(round_state['seats'])
        
        # Professional position classification
        # 0 = early, 1 = middle, 2 = late, 3 = blinds
        if num_players <= 3:
            if (my_position - dealer_position) % num_players == 0:  # Button
                position_class = 2  # Late
            else:  # Blinds
                position_class = 3  # Blinds
        else:
            pos_relative = (my_position - dealer_position) % num_players
            if pos_relative == 0:  # Button
                position_class = 2  # Late
            elif pos_relative <= 2:  # SB, BB
                position_class = 3  # Blinds
            elif pos_relative < num_players / 3 + 2:  # First third of table after blinds
                position_class = 0  # Early
            elif pos_relative < 2 * num_players / 3 + 2:  # Second third
                position_class = 1  # Middle
            else:  # Last third
                position_class = 2  # Late
                
        self.state += (position_class,)
        
        # Calculate Stack-to-Pot Ratio (SPR) - key concept in professional poker
        # SPR = (My Stack) / (Pot Size)
        if main_pot > 0:
            spr = my_stack / main_pot
        else:
            spr = 10.0  # Default high value when pot is empty
            
        self.state += (spr,)
        
        # Add pot odds if facing a bet
        pot_odds = 0.0
        for street in ['preflop', 'flop', 'turn', 'river']:
            if street in round_state['action_histories'] and round_state['action_histories'][street]:
                actions = round_state['action_histories'][street]
                # Find the last bet/raise that isn't ours
                for action in reversed(actions):
                    if action['uuid'] != self.uuid and action['action'] in ['call', 'raise'] and action.get('amount', 0) > 0:
                        amount_to_call = action.get('amount', 0)
                        if amount_to_call > 0:
                            pot_odds = amount_to_call / (main_pot + amount_to_call)
                        break
                        
        self.state += (pot_odds,)
        
        # Add opponent modeling features if available
        if self.opponent_tracker is not None:
            all_positions = self.opponent_tracker._get_positions(round_state)
            
            # Add features for each opponent
            for i, seat in enumerate(round_state['seats']):
                if i != self.player_id:  # Skip ourselves
                    opp_features = self.opponent_tracker.get_opponent_feature_vector(i)
                    
                    # Add position context to features
                    opp_position = all_positions[i]
                    opp_position_class = self.opponent_tracker._position_to_type(opp_position, num_players)
                    
                    # Get exploit strategy for this opponent
                    exploit_strategy = self.opponent_tracker.get_exploit_strategy(
                        i, position_class, self.position_type_to_class(opp_position_class))
                        
                    # Add strategy adjustments to state
                    self.state += tuple(opp_features)
                    self.state += (exploit_strategy['value_bet_sizing'], 
                                  exploit_strategy['bluff_frequency'],
                                  exploit_strategy['call_threshold'])
                else:
                    # Add placeholder values for our own position
                    self.state += tuple([0.0] * 10)  # Zero features for self
                    self.state += (1.0, 1.0, 1.0)  # Neutral strategy adjustments
                    
        return self.state
        
    def position_type_to_class(self, position_type):
        """Convert position type string to class constant"""
        mapping = {
            'early': EARLY_POSITION,
            'middle': MIDDLE_POSITION,
            'late': LATE_POSITION,
            'blinds': BLINDS
        }
        return mapping.get(position_type, MIDDLE_POSITION)
    
    def declare_action(self, valid_actions, hole_card, round_state):
        """Declare an action based on NFSP strategy with professional poker adjustments"""
        self.episode += 1
        
        # Preprocess state
        hole_card_1 = self.card_to_int(hole_card[0])
        hole_card_2 = self.card_to_int(hole_card[1])
        self.hole_card = (hole_card_1, hole_card_2)
        community_card = self.community_card_to_tuple(round_state['community_card'])
        
        # Track opponent actions before making our decision
        if self.opponent_tracker is not None:
            self.opponent_tracker.update_from_game_state(round_state, self.player_id, round_state["street"])
        
        # Get enhanced state with opponent modeling
        self.state = self.get_state(community_card, round_state)
        self.state = self.process_state(self.state)
        
        # Store current state for later
        self.current_state = self.state
        
        # Process valid actions and pot size
        pot_size = round_state['pot']['main']['amount']
        side_pots = round_state['pot'].get('side', [])
        for side_pot in side_pots:
            pot_size += side_pot['amount']
            
        # Calculate Stack-to-Pot Ratio (SPR) - crucial for professional betting strategy
        my_stack = round_state['seats'][self.player_id]['stack']
        spr = my_stack / max(1, pot_size)  # Avoid division by zero
        
        # Get my position for position-aware play
        dealer_position = round_state['dealer_btn']
        for player in round_state['seats']:
            if player['uuid'] == self.uuid:
                my_position = int(player['name'][-1])
                break
                
        relative_position = (my_position - dealer_position) % len(round_state['seats'])
        position_type = self.get_position_type(relative_position, len(round_state['seats']))
        
        # Adjust raise sizes based on position and SPR (professional concept)
        if spr < self.spr_thresholds['commitment']:
            # With low SPR, focus on all-in or fold decisions with simplified bet sizing
            self.raisesizes = [0.5, 1.0, 2.0] 
        elif spr < self.spr_thresholds['postflop_play']:
            # Medium SPR, standard bet sizing
            self.raisesizes = [0.5, 0.75, 1.0, 1.5, 2.0]
        else:
            # High SPR, more flexible bet sizing including small and large bets
            self.raisesizes = [0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
            
        # Position-based adjustments to raise sizes (pros use different sizings by position)
        position_adjust = self.position_adjustments[position_type]
        
        # In early position, pros tend to use larger sizings with stronger ranges
        if position_type == 'early' and round_state["street"] == 'preflop':
            self.raisesizes = [size for size in self.raisesizes if size >= 0.75]
        # In late position, more flexibility including smaller raises for stealing
        elif position_type == 'late' and round_state["street"] == 'preflop':
            if not any(a['action'] == 'raise' for a in round_state['action_histories'].get('preflop', [])):
                # Add small steal raises when opening in late position
                self.raisesizes = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0]
                
        # Prepare raise amounts with professional sizing considerations
        for action in valid_actions:
            if action['action'] == 'raise':
                min_raise = action['amount']['min']
                max_raise = action['amount']['max']
                
                for raisesize in self.raisesizes:
                    # Pros often size their bets relative to pot
                    raiseamount = raisesize * pot_size
                    if raiseamount > min_raise and raiseamount < max_raise:
                        action['amount'][raisesize] = raiseamount
                    elif raiseamount <= min_raise: 
                        action['amount'][raisesize] = min_raise
                    elif raiseamount >= max_raise: 
                        action['amount'][raisesize] = max_raise
                        
                # Special case for all-in considerations with low SPR
                if spr < 3.0 and max_raise < my_stack:
                    action['amount']['allin'] = my_stack
                        
                amounts = action['amount']
                # Sort the amounts
                min_value = {'min': amounts.pop('min')}
                max_value = {'max': amounts.pop('max')}
                sorted_amount = dict(sorted(amounts.items(), key=lambda item: item[1]))
                final_amount = {**min_value, **sorted_amount, **max_value}
                action['amount'] = final_amount
        
        # Select action using NFSP
        action_index = self.select_action(self.state, valid_actions)
        
        # Convert to API format (fold, call, or raise)
        if action_index > 1:
            amount = list(valid_actions[2]["amount"].items())[action_index - 2][1]
            action = "raise"
        elif action_index == 1:
            amount = valid_actions[1]["amount"]
            action = "call"
        else:
            amount = 0
            action = "fold"
            
        # Track poker statistics
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
                    for previous_action in reversed(pre_flop_action[:-1]):
                        if previous_action["action"].lower() == "raise":
                            self.three_bet += 1
                            self.last_3_bet_action = action
                            break
                            
        # Record action for statistics
        stats_action = action
        if action == "call" and amount == 0:
            stats_action = 'check'
            
        if not self.street_first_action:
            self.action_stat[round_state["street"]][stats_action] += 1 
            self.card_action_stat[round_state["street"]][stats_action][self.hand_type] += 1
            self.street_first_action = stats_action
            if round_state["street"] == 'preflop':
                self.hand_count += 1
                
        # Store action for learning
        self.history.append((self.state, action_index))
                
        return action, math.floor(amount)
    
    def receive_game_start_message(self, game_info):
        """Initialize game information and opponent modeling"""
        self.nb_player = game_info['player_num']
        for i in range(0, len(game_info['seats'])):
            if self.uuid == game_info['seats'][i]['uuid']:
                self.player_id = i
                break
        self.stack = 100
        
        # Initialize opponent tracker with number of players in the game
        if self.opponent_tracker is None:
            self.opponent_tracker = OpponentTracker(self.nb_player)
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        """Initialize round information"""
        self.last_vpip_action = None
        self.last_pfr_action = None
        self.last_3_bet_action = None
        self.hand_type = self.get_hand_type(hole_card)
        self.best_response_mode = None  # Will be set on first action
        self.history = []
    
    def receive_street_start_message(self, street, round_state):
        """Initialize street information"""
        self.street_first_action = None
    
    def receive_game_update_message(self, new_action, round_state):
        """Process game updates"""
        pass
    
    @staticmethod
    def round_int_to_string(round_int):
        """Convert round integer to string"""
        m = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
        return m[round_int]
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        """Process round results, update networks, and track opponent shown hands"""
        if not self.training or len(self.history) < 1:
            return
            
        # Calculate reward
        for players in round_state['seats']: 
            if players['uuid'] == self.uuid:
                reward = players['stack'] - self.stack
                self.stack = players['stack']
                break
                
        reward /= 100  # Normalize reward
        self.accumulated_reward += reward
        
        # Record the terminal state
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
        terminal_state = self.process_state(self.state)
        
        # Store experiences and calculate TD errors
        for i in range(len(self.history)):
            state, action = self.history[i]
            
            if i == len(self.history) - 1:
                # Terminal transition
                next_state = None
            else:
                next_state = self.history[i+1][0]
                
            # Store transition in BR memory if using best response
            if self.best_response_mode:
                self.store_transition(state, action, reward, next_state)
        
        # Update NFSP networks
        self.update()
        
        # Save model periodically
        if self.episode % 1000 == 0:
            self.save_model()
            
        # Update card reward statistics
        self.card_reward_stat[self.hand_type] += reward
        
        # Track opponent hands shown at showdown
        if self.opponent_tracker is not None and hand_info:
            for player_info in hand_info:
                player_uuid = player_info.get('uuid')
                hole_card = player_info.get('hole_card')
                
                if player_uuid and hole_card and player_uuid != self.uuid:
                    # Find player index
                    for i, seat in enumerate(round_state['seats']):
                        if seat['uuid'] == player_uuid:
                            player_idx = i
                            break
                    else:
                        continue  # Player not found
                        
                    # Record hand type
                    hand_type = self.get_hand_type(hole_card)
                    self.opponent_tracker.player_stats[player_idx]['hand_types_shown'][hand_type] += 1

    def get_position_type(self, relative_position, num_players):
        """Convert relative position to position type"""
        if num_players <= 3:
            if relative_position == 0:  # Button
                return 'late'
            else:  # Blinds
                return 'blinds'
        else:
            if relative_position == 0:  # Button
                return 'late'
            elif relative_position <= 2:  # SB, BB
                return 'blinds'
            elif relative_position < num_players / 3 + 2:  # First third of table after blinds
                return 'early'
            elif relative_position < 2 * num_players / 3 + 2:  # Second third
                return 'middle'
            else:  # Last third
                return 'late'

def rank_value(rank):
    """Helper function to convert card rank to numeric value"""
    values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
              '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    return values[rank] 