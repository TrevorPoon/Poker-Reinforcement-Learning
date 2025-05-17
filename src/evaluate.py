import argparse
import os
import torch
import torch.optim as optim # Required for PPOPlayer instantiation
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random
import logging

from pypokerengine.api.game import setup_config, start_poker

from my_players.PPOPlayer import PPOPlayer, ActorCriticNetwork
from my_players.PPOPlayer import card_to_int, community_card_to_tuple_indices, get_hand_type_str_static
from my_players.HonestPlayer_v2 import HonestPlayer
from my_players.AllCall import AllCallPlayer
from utils import Charts # Assuming Charts.py is in utils

# --- Global list to store decision data from the evaluated agent ---
evaluation_log = []

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# --- Helper function to define a unique spot key ---
def get_spot_key(street, community_cards, action_history_on_street):
    """
    Generates a key for a decision spot.
    Example: PREFLOP_[]_R_C (Raise, Call)
    FLOP_[H2_D3_S4]_B_R (Bet, Raise)
    """
    board_str = "_".join(sorted(community_cards)) if community_cards else "[]"
    
    # Summarize action history for the current street
    # For simplicity, just take the types of actions (R, C, B, K, F)
    action_summary = []
    if street.lower() in action_history_on_street:
        for action_item in action_history_on_street[street.lower()]:
            action_type = action_item['action']
            if action_type == "RAISE":
                action_summary.append("R")
            elif action_type == "CALL":
                action_summary.append("C")
            elif action_type == "BET": # pypokerengine uses BET for first bet on postflop
                action_summary.append("B")
            # Add CHECK ('K') and FOLD ('F') if needed, though agent acts before folding usually
            # For now, focusing on aggressive/passive actions leading to agent's decision.
    
    actions_str = "_".join(action_summary)
    return f"{street.upper()}_{board_str}_{actions_str}"


# --- EvaluationPPOPlayer Class ---
class EvaluationPPOPlayer(PPOPlayer):
    def __init__(self, actor_critic_net, optimizer, model_save_path, optimizer_save_path,
                 initial_stack, small_blind,
                 num_feats, num_actions, lstm_hidden_size, card_embedding_dim, num_card_indices_in_state,
                 eval_log_list): # Added eval_log_list

        super().__init__(shared_actor_critic_net=actor_critic_net, shared_optimizer=optimizer,
                         model_save_path=model_save_path, optimizer_save_path=optimizer_save_path,
                         training=False, # IMPORTANT: Set training to False
                         initial_stack=initial_stack, small_blind=small_blind,
                         num_feats=num_feats, num_actions=num_actions, lstm_hidden_size=lstm_hidden_size,
                         card_embedding_dim=card_embedding_dim, num_card_indices_in_state=num_card_indices_in_state)
        self.eval_log_list = eval_log_list # Store the global list reference

    def declare_action(self, valid_actions, hole_card_pokerlib, round_state):
        # --- Standard PPOPlayer state processing and network forward pass ---
        # (This part is largely similar to PPOPlayer's declare_action)
        hole_card_idx1 = card_to_int(hole_card_pokerlib[0])
        hole_card_idx2 = card_to_int(hole_card_pokerlib[1])
        self.hole_card_tuple_raw_indices = (hole_card_idx1, hole_card_idx2)
        community_card_indices_tuple = community_card_to_tuple_indices(round_state['community_card'])

        raw_state_tuple = self.get_state(community_card_indices_tuple, round_state)
        current_processed_state = self.process_state(raw_state_tuple)
        
        state_list_for_net = list(current_processed_state)
        expected_len = self.num_feats
        if len(state_list_for_net) < expected_len:
            state_list_for_net.extend([0.0] * (expected_len - len(state_list_for_net)))
        elif len(state_list_for_net) > expected_len:
            state_list_for_net = state_list_for_net[:expected_len]
        
        state_tensor = torch.tensor([state_list_for_net], device=self.device, dtype=torch.float)

        action_probs_tensor, state_value_tensor, next_lstm_hidden = torch.Tensor([0]), torch.Tensor([0]), None # Placeholder
        with torch.no_grad(): # Ensure no gradients are computed during evaluation
            action_probs_tensor, state_value_tensor, next_lstm_hidden = self.actor_critic_net(state_tensor, self.current_lstm_hidden)
        
        self.current_lstm_hidden = next_lstm_hidden # Update LSTM hidden state

        # --- Action selection logic (copied and adapted from PPOPlayer) ---
        # Create a mask for valid actions
        valid_action_mask = np.zeros(self.num_actions, dtype=bool)
        valid_action_mask[0] = any(va['action'] == 'fold' for va in valid_actions)
        valid_action_mask[1] = any(va['action'] == 'call' for va in valid_actions)
        can_raise = any(va['action'] == 'raise' and va['amount']['min'] != -1 for va in valid_actions)
        if can_raise:
            valid_action_mask[2:] = True

        masked_action_probs = action_probs_tensor.cpu().detach().numpy().flatten()
        masked_action_probs[~valid_action_mask] = 0.0
        
        if np.sum(masked_action_probs) == 0:
            if valid_action_mask[0]: masked_action_probs[0] = 1.0
            elif valid_action_mask[1]: masked_action_probs[1] = 1.0
            else:
                 idx = np.where(valid_action_mask)[0]
                 if len(idx)>0 : masked_action_probs[idx[0]] = 1.0
                 else: masked_action_probs[0] = 1.0
        
        prob_sum = np.sum(masked_action_probs)
        final_probs = masked_action_probs / prob_sum if prob_sum > 0 else np.zeros_like(masked_action_probs)
        if np.sum(final_probs) == 0: final_probs[0] = 1.0 # Ultimate fallback to fold

        # In evaluation, we might want the agent to play deterministically (highest prob action)
        # or sample. For GTO-like analysis, sampling according to policy is better.
        # chosen_action_idx = np.argmax(final_probs) # Deterministic
        
        # Sample action based on probabilities for evaluation to see strategy distribution
        # Need to handle case where final_probs might sum to 0 if all masked out, though fallback handles it.
        if np.all(final_probs == 0): # Should be prevented by fallback
            chosen_action_idx = 0 # Fold
        else:
            chosen_action_idx = np.random.choice(np.arange(self.num_actions), p=final_probs)


        # --- Map action_index to game action string and amount (from PPOPlayer) ---
        chosen_action_str, chosen_amount = "fold", 0
        if chosen_action_idx == 1: # Call/Check
            chosen_action_str = "call"
            call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
            if call_action: chosen_amount = call_action["amount"]
        elif chosen_action_idx > 1: # Raise
            chosen_action_str = "raise"
            raise_main_action = next((va for va in valid_actions if va['action'] == 'raise'), None)
            if raise_main_action and raise_main_action['amount']['min'] != -1 :
                pot_size = round_state['pot']['main']['amount']
                min_r = raise_main_action['amount']['min']
                max_r = raise_main_action['amount']['max']
                raise_option_idx = chosen_action_idx - 2
                if raise_option_idx < len(self.raisesizes):
                    val = self.raisesizes[raise_option_idx] * pot_size
                    if min_r != -1 and val < min_r: val = min_r
                    if max_r != -1 and val > max_r: val = max_r
                    chosen_amount = int(val) # Using int, pypokerengine uses int
                else: chosen_amount = min_r
                if chosen_amount < min_r and min_r != -1: chosen_amount = min_r
                if chosen_amount > max_r and max_r != -1: chosen_amount = max_r
                if chosen_amount == 0 or (min_r != -1 and chosen_amount < min_r):
                    if valid_action_mask[1]:
                        chosen_action_str, chosen_amount = "call", next(va for va in valid_actions if va['action'] == 'call')['amount']
                    else: chosen_action_str, chosen_amount = "fold", 0
            else:
                if valid_action_mask[1]:
                    chosen_action_str, chosen_amount = "call", next(va for va in valid_actions if va['action'] == 'call')['amount']
                else: chosen_action_str, chosen_amount = "fold", 0
        
        # --- Log data for evaluation ---
        log_entry = {
            "player_uuid": self.uuid,
            "hole_cards": hole_card_pokerlib,
            "community_cards": round_state['community_card'],
            "street": round_state['street'],
            "action_history_full": round_state['action_histories'], # Full history for context
            "betting_history_this_street": round_state['action_histories'].get(round_state['street'].lower(), []),
            "valid_actions": valid_actions, # For context
            "chosen_action_str": chosen_action_str,
            "chosen_action_amount": chosen_amount,
            "chosen_action_idx": chosen_action_idx,
            "action_probabilities": final_probs.tolist(), # Log the final, masked, normalized probabilities
            "estimated_value": state_value_tensor.item(), # Critic's V(s) as EV proxy
            "round_pot_before_action": round_state['pot']['main']['amount'] + sum(p['amount'] for p in round_state['pot'].get('side',[])),
            "hand_reward": 0.0 # Will be updated at end of hand
        }
        # Add player position relative to button
        num_game_players = len(round_state['seats'])
        dealer_btn_seat_idx = round_state['dealer_btn']
        my_seat_idx_in_game = self.player_id # This is the absolute seat index
        log_entry["player_position_abs"] = my_seat_idx_in_game
        log_entry["player_position_rel_button"] = (my_seat_idx_in_game - dealer_btn_seat_idx + num_game_players) % num_game_players
        
        # Use the more detailed spot key generation
        log_entry["spot_key"] = get_spot_key(
            round_state['street'], 
            round_state['community_card'], 
            round_state['action_histories'] # Pass full history
        )

        # Store a temporary reference to this log entry to update reward later
        self.current_hand_decision_logs.append(log_entry) 
        self.eval_log_list.append(log_entry) # Append to the global list

        return chosen_action_str, int(chosen_amount) # Ensure amount is int

    def receive_round_start_message(self, round_count, hole_card, seats):
        super().receive_round_start_message(round_count, hole_card, seats)
        self.current_hand_decision_logs = [] # Track decisions for current hand for reward assignment

    def receive_round_result_message(self, winners, hand_info, round_state):
        # Original PPOPlayer might do training updates here if training=True
        # Since we are in eval mode (training=False), super call is mostly for state update
        super().receive_round_result_message(winners, hand_info, round_state)

        # Calculate actual reward for the hand for the evaluated player
        final_stack = 0
        initial_stack_this_hand = -1

        for player_seat_info in round_state['seats']:
            if player_seat_info['uuid'] == self.uuid:
                final_stack = player_seat_info['stack']
                # Try to find initial stack for this hand from hand_info if available
                # hand_info structure: { 'dealer_btn': 0, 'hand': {'hole_card': [], 'hand_strength': -1}, 'round_count': 1}
                # The hand_info passed here might not have initial stack for each player for THIS hand easily.
                # We stored self.stack at the start of PPOPlayer, which is initial_stack for the game.
                # We need stack at the START of this specific hand.
                # PPOPlayer has self.stack which is updated. Let's use a snapshot.
                # This is complex because self.stack in PPOPlayer is dynamic.
                # For simplicity, let's assume the reward is (change in stack during this hand) / game_initial_stack
                # This was `reward_for_hand = (final_stack - self.stack_at_hand_start) / self.initial_stack`
                # Let's assume self.stack was correctly snapshotted at start of hand by superclass or here.
                # PPOPlayer's own self.stack tracks current stack.
                # The reward logic in PPOPlayer uses self.initial_stack (game initial) for normalization.
                # We need the change from the start of *this* hand.
                # The `hand_info` in `pypokerengine` is about the hand itself (winner, cards), not player stack changes during it.
                # `round_state` has final stacks. We need initial stacks *for this hand*.

                # A simple way: calculate reward based on chips won/lost compared to what was bet.
                # This is also tricky. `pypokerengine`'s game_result is simpler.
                # For now, let's record the raw win/loss in chips for the hand.
                
                # Find player in winners list
                player_won_amount = 0
                for winner_info in winners:
                    if winner_info['uuid'] == self.uuid:
                        player_won_amount = winner_info['stack'] # This is their stack AFTER winning
                        # This isn't just "amount won", it's their new total stack.
                        # Amount won = new_total_stack - stack_before_pot_distribution
                
                # For now, let's use the logic similar to PPOPlayer: (final_stack_of_hand - initial_stack_of_hand)
                # PPOPlayer's self.stack is updated to final_stack at the end of its receive_round_result.
                # So, `final_stack - self.stack_before_this_update` would be the change for this hand.
                # The `self.stack` in PPOPlayer is the stack _before_ this hand's result is applied.
                # So, `final_stack - self.stack` IS the profit/loss from this hand.
                
                hand_profit_loss = final_stack - self.stack # self.stack is from before this hand's outcome
                normalized_reward = hand_profit_loss / self.initial_stack # Normalize by game initial stack

                for log_entry in self.current_hand_decision_logs:
                    if log_entry["player_uuid"] == self.uuid:
                        log_entry["hand_reward"] = normalized_reward # Assign to all decisions in this hand
                break
        
        self.current_hand_decision_logs = [] # Clear for next hand

# --- Plotting ---
def plot_action_pie_chart(action_freq_dict, spot_key_str, title_prefix_str, output_dir_path):
    """Generates and saves an action frequency pie chart."""
    labels = []
    sizes = []
    colors_list = []
    
    # Use a consistent order and colors from Charts.ACTIONS_ORDER and Charts.ACTION_COLORS
    for action_name in Charts.ACTIONS_ORDER:
        if action_name in action_freq_dict and action_freq_dict[action_name] > 0:
            labels.append(f"{action_name.capitalize()} ({action_freq_dict[action_name]})")
            sizes.append(action_freq_dict[action_name])
            colors_list.append(Charts.ACTION_COLORS.get(action_name, '#CCCCCC')) # Default color if not found

    if not sizes:
        logger.warning(f"No action frequencies to plot for pie chart (spot: {spot_key_str})")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
    ax.axis('equal')
    ax.set_title(f"{title_prefix_str}: Action Frequencies for Spot: {spot_key_str}")
    
    try:
        filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{spot_key_str}_action_pie.png")
        fig.savefig(filepath)
        logger.info(f"Saved pie chart to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pie chart for spot {spot_key_str}: {e}")
    finally:
        plt.close(fig) # Ensure figure is closed

# --- Main Evaluation Logic ---
def run_evaluation(model_path, num_hands, output_dir, initial_stack_val, small_blind_val, max_rounds_val):
    global evaluation_log # Clear previous log if any
    evaluation_log = []

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting evaluation for model: {model_path}, Hands: {num_hands}")
    logger.info(f"Output will be saved to: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PPO Agent Network Parameters (from training.py defaults for PPO) ---
    ppo_num_feats = 85
    ppo_num_actions = 11
    ppo_lstm_hidden_size = 128
    ppo_card_embedding_dim = 16
    ppo_num_card_indices_in_state = 7
    # PPO_LEARNING_RATE_ACTOR_CRITIC = 1e-4 # Not needed for optimizer in eval if not updated

    # Initialize ActorCriticNetwork
    actor_critic_net = ActorCriticNetwork(
        input_raw_feat_dim=ppo_num_feats,
        num_actions=ppo_num_actions,
        lstm_hidden_size=ppo_lstm_hidden_size,
        card_embedding_dim=ppo_card_embedding_dim,
        num_card_indices=ppo_num_card_indices_in_state
    ).to(device)

    # Load the trained model weights
    try:
        actor_critic_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return

    actor_critic_net.eval() # Set network to evaluation mode

    # Optimizer: PPOPlayer expects an optimizer. Create a dummy one if not training.
    # The learning rate doesn't matter if not used.
    dummy_optimizer = optim.Adam(actor_critic_net.parameters(), lr=1e-4) 
    # Dummy paths for optimizer save/load, PPOPlayer expects them
    dummy_optimizer_path = os.path.join(output_dir, "dummy_ppo_optimizer.pt")


    # Initialize the PPO player for evaluation
    # Pass the global `evaluation_log` list to the player instance
    ppo_eval_player = EvaluationPPOPlayer(
        actor_critic_net=actor_critic_net,
        optimizer=dummy_optimizer, # Pass dummy optimizer
        model_save_path=model_path, # model_path is for loading, not saving here
        optimizer_save_path=dummy_optimizer_path, # Dummy path
        initial_stack=initial_stack_val,
        small_blind=small_blind_val,
        num_feats=ppo_num_feats,
        num_actions=ppo_num_actions,
        lstm_hidden_size=ppo_lstm_hidden_size,
        card_embedding_dim=ppo_card_embedding_dim,
        num_card_indices_in_state=ppo_num_card_indices_in_state,
        eval_log_list=evaluation_log # Pass the global list here
    )

    # --- Game Setup (similar to training.py) ---
    # For evaluation, the PPO agent plays against baseline agents (e.g., HonestPlayer)
    # We want to evaluate ONE PPO agent.
    
    # Simulation loop
    for i in range(num_hands):
        if (i + 1) % 100 == 0:
            logger.info(f"Simulating hand {i+1}/{num_hands}")

        config = setup_config(max_round=max_rounds_val, initial_stack=initial_stack_val, small_blind_amount=small_blind_val)
        
        # Register PPO agent and 5 opponents
        players_for_game = [ppo_eval_player]
        for _ in range(5): # Add 5 opponents
            players_for_game.append(HonestPlayer()) # Or AllCallPlayer, or mix
        
        random.shuffle(players_for_game) # Shuffle positions

        for player_idx, player_algo in enumerate(players_for_game):
            config.register_player(name=f"p{player_idx+1}", algorithm=player_algo)
        
        try:
            game_result = start_poker(config, verbose=0)
            # Logging is done within EvaluationPPOPlayer's declare_action and receive_round_result
        except Exception as e:
            logger.error(f"Error during game simulation at hand {i+1}: {e}")
            # Optionally, decide if to continue or stop evaluation
            # For now, log and continue
            if i > 0 and (i+1) % 10 ==0: # to avoid spamming
                 logger.info("Continuing simulation despite error in one hand.")

    logger.info(f"Finished {num_hands} simulation hands. Total decisions logged: {len(evaluation_log)}")
    process_evaluation_log(evaluation_log, output_dir, "PPO_Eval")


def process_evaluation_log(log_data, output_dir_path, title_prefix_str):
    if not log_data:
        logger.warning("No evaluation data logged. Skipping processing.")
        return

    # Group decisions by spot_key
    spots_data = {}
    for decision_log_entry in log_data:
        spot_key = decision_log_entry['spot_key']
        if spot_key not in spots_data:
            spots_data[spot_key] = []
        spots_data[spot_key].append(decision_log_entry)

    logger.info(f"Processing {len(spots_data)} unique spots.")

    for spot_key_str, decisions_in_spot_list in spots_data.items():
        if not decisions_in_spot_list: continue

        logger.info(f"Analyzing spot: {spot_key_str} ({len(decisions_in_spot_list)} decisions)")

        # Aggregate data for this spot
        # spot_hand_action_freq: hand_str -> {'action': count}
        # spot_hand_ev: hand_str -> {'total_ev': float, 'count': int}
        # spot_action_overall_freq: {'action': count}
        spot_hand_action_freq_map = {} 
        spot_hand_ev_map = {} 
        spot_action_overall_freq_dict = {action: 0 for action in Charts.ACTIONS_ORDER}


        for decision in decisions_in_spot_list:
            try:
                hand_type_str = get_hand_type_str_static(decision['hole_cards'])
            except Exception as e:
                logger.error(f"Could not get hand type for {decision['hole_cards']}: {e}")
                hand_type_str = "UnknownHand"

            action_taken = decision['chosen_action_str']
            if action_taken == "call" and decision['chosen_action_amount'] == 0:
                action_taken = "check" # Normalize call 0 to check for stats

            # Ensure action_taken is one of the standard ones for aggregation
            if action_taken not in spot_action_overall_freq_dict:
                # If action_taken (e.g. 'bet') is not in Charts.ACTIONS_ORDER, map or ignore
                # For simplicity, let's assume chosen_action_str is already one of fold, call, raise
                # 'check' is handled above. 'bet' from pypokerengine would map to 'raise' if it's an opening bet.
                # This part needs careful mapping if PPOPlayer can output other action strings.
                # Assuming 'fold', 'call', 'raise', 'check' are the primary outcomes.
                pass # Or log a warning

            # 1. For GTO-style Range Grid (action distribution per hand)
            if hand_type_str not in spot_hand_action_freq_map:
                spot_hand_action_freq_map[hand_type_str] = {action: 0 for action in Charts.ACTIONS_ORDER}
            if action_taken in spot_hand_action_freq_map[hand_type_str]:
                 spot_hand_action_freq_map[hand_type_str][action_taken] += 1

            # 2. For EV Heatmap (using estimated_value from critic or actual hand_reward)
            if hand_type_str not in spot_hand_ev_map:
                spot_hand_ev_map[hand_type_str] = {'total_ev': 0.0, 'count': 0, 'total_hand_reward': 0.0}
            
            spot_hand_ev_map[hand_type_str]['total_ev'] += decision['estimated_value'] # Critic's V(s)
            spot_hand_ev_map[hand_type_str]['total_hand_reward'] += decision['hand_reward'] # Actual normalized hand reward
            spot_hand_ev_map[hand_type_str]['count'] += 1
            
            # 3. For Overall Action Frequency Pie Chart
            if action_taken in spot_action_overall_freq_dict:
                 spot_action_overall_freq_dict[action_taken] += 1
        
        # --- Generate and Save Charts for this spot ---

        # 1. GTO Style Action Grid (shows action distribution per hand)
        # Transform spot_hand_action_freq_map for Charts.plot_gto_style_action_grid
        # Expected: card_action_stat_for_plot = { 'action': { 'hand_type': count } }
        card_action_stat_for_gto_plot = {action: {} for action in Charts.ACTIONS_ORDER}
        for hand_str_key, action_counts_dict in spot_hand_action_freq_map.items():
            for action_name_key, count_val in action_counts_dict.items():
                if count_val > 0: # Only add if count > 0
                    card_action_stat_for_gto_plot[action_name_key][hand_str_key] = count_val
        
        # Using street=spot_key_str for title uniqueness, player_num=None as it's one agent
        fig_gto_grid = Charts.plot_gto_style_action_grid(card_action_stat_for_gto_plot, spot_key_str, title_prefix_str, player_num=None)
        if fig_gto_grid:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{spot_key_str}_GTO_action_grid.png")
                fig_gto_grid.savefig(filepath)
                logger.info(f"Saved GTO action grid to {filepath}")
            except Exception as e:
                logger.error(f"Error saving GTO grid for spot {spot_key_str}: {e}")
            finally:
                plt.close(fig_gto_grid)

        # 2. Action Frequency Pie Chart
        plot_action_pie_chart(spot_action_overall_freq_dict, spot_key_str, title_prefix_str, output_dir_path)

        # 3. EV Heatmap (using critic's estimated value)
        ev_heatmap_data_critic = {
            hand_str_key: data_dict['total_ev'] / data_dict['count'] if data_dict['count'] > 0 else 0
            for hand_str_key, data_dict in spot_hand_ev_map.items() if data_dict['count'] > 0
        }
        fig_ev_heatmap_critic = Charts.plot_hand_reward_heatmap(ev_heatmap_data_critic, f"{spot_key_str}_CriticEV", title_prefix_str, player_num=None)
        if fig_ev_heatmap_critic:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{spot_key_str}_EV_heatmap_critic.png")
                fig_ev_heatmap_critic.savefig(filepath)
                logger.info(f"Saved Critic EV heatmap to {filepath}")
            except Exception as e:
                logger.error(f"Error saving Critic EV heatmap for spot {spot_key_str}: {e}")
            finally:
                plt.close(fig_ev_heatmap_critic)

        # 3b. EV Heatmap (using actual hand reward)
        ev_heatmap_data_actual = {
            hand_str_key: data_dict['total_hand_reward'] / data_dict['count'] if data_dict['count'] > 0 else 0
            for hand_str_key, data_dict in spot_hand_ev_map.items() if data_dict['count'] > 0
        }
        fig_ev_heatmap_actual = Charts.plot_hand_reward_heatmap(ev_heatmap_data_actual, f"{spot_key_str}_ActualReward", title_prefix_str, player_num=None)
        if fig_ev_heatmap_actual:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{spot_key_str}_EV_heatmap_actual.png")
                fig_ev_heatmap_actual.savefig(filepath)
                logger.info(f"Saved Actual Reward heatmap to {filepath}")
            except Exception as e:
                logger.error(f"Error saving Actual Reward heatmap for spot {spot_key_str}: {e}")
            finally:
                plt.close(fig_ev_heatmap_actual)


# --- Argparse and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained PPO Poker Agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.pt file).")
    parser.add_argument("--num_hands", type=int, default=10000, help="Number of hands to simulate for evaluation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation charts. Defaults to 'evaluation_results/model_name_timestamp'.")
    parser.add_argument("--initial_stack", type=int, default=100, help="Initial chip stack for players.")
    parser.add_argument("--small_blind", type=float, default=0.5, help="Small blind amount.")
    parser.add_argument("--max_rounds", type=int, default=36, help="Max rounds per game (pypokerengine game setting).")


    args = parser.parse_args()

    if args.output_dir is None:
        model_name_no_ext = os.path.splitext(os.path.basename(args.model_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join("evaluation_results", f"{model_name_no_ext}_{timestamp}")
    
    # Add file handler for logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "evaluation_log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    run_evaluation(args.model_path, args.num_hands, args.output_dir, 
                   args.initial_stack, args.small_blind, args.max_rounds)

    logger.info("Evaluation script finished.") 