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
    
    action_summary = []
    # Process actions on the current street that happened *before* the agent's decision
    current_street_history = action_history_on_street.get(street.lower(), [])
    for action_item in current_street_history:
        action_type = action_item['action']
        amount = action_item.get('amount', 0)
        paid = action_item.get('paid', 0) # For blinds

        if action_type == "SMALLBLIND":
            action_summary.append("SB")
        elif action_type == "BIGBLIND":
            action_summary.append("BB")
        elif action_type == "ANTE":
            action_summary.append("A")
        elif action_type == "FOLD":
            action_summary.append("F")
        elif action_type == "CALL":
            if amount == 0: # Could be a check or calling a 0 amount bet
                 # Check if it's a call of a previous bet or a check.
                 # PyPokerEngine uses CALL amount 0 for check.
                action_summary.append("K") # Check
            else:
                action_summary.append("C") # Call
        elif action_type == "BET": # Opening bet on flop, turn, river
            action_summary.append("B")
        elif action_type == "RAISE":
            action_summary.append("R")
    
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

        action_probs_tensor, state_value_tensor, next_lstm_hidden = None, None, None
        with torch.no_grad(): # Ensure no gradients are computed during evaluation
            action_probs_tensor, state_value_tensor, next_lstm_hidden = self.actor_critic_net(state_tensor, self.current_lstm_hidden)
        
        self.current_lstm_hidden = next_lstm_hidden # Update LSTM hidden state

        # --- Action selection logic (copied and adapted from PPOPlayer) ---
        # Create a mask for valid actions. This is crucial for correctness.
        valid_action_mask = np.zeros(self.num_actions, dtype=bool)
        valid_action_mask[0] = any(va['action'] == 'fold' for va in valid_actions)
        valid_action_mask[1] = any(va['action'] == 'call' for va in valid_actions) # This includes check (call amount 0)
        can_raise = any(va['action'] == 'raise' and va['amount']['min'] != -1 for va in valid_actions)
        if can_raise:
            valid_action_mask[2:] = True

        masked_action_probs = action_probs_tensor.cpu().detach().numpy().flatten()
        masked_action_probs[~valid_action_mask] = 0.0
        
        # Fallback if all masked probabilities are zero (e.g. if network outputs near zero for all valid actions)
        if np.sum(masked_action_probs) == 0:
            if valid_action_mask[0]: masked_action_probs[0] = 1.0 # Prioritize fold
            elif valid_action_mask[1]: masked_action_probs[1] = 1.0 # Prioritize call/check
            else: # Should not happen if fold or call/check is almost always an option
                 idx = np.where(valid_action_mask)[0]
                 if len(idx)>0 : masked_action_probs[idx[0]] = 1.0 # Pick first available valid action
                 else: masked_action_probs[0] = 1.0 # Ultimate fallback: fold (should be extremely rare)
        
        prob_sum = np.sum(masked_action_probs)
        final_probs = masked_action_probs / prob_sum if prob_sum > 0 else np.zeros_like(masked_action_probs)
        if np.sum(final_probs) == 0: final_probs[0] = 1.0 # Ensure probabilities sum to 1, default to fold if error

        # Sample action based on probabilities for evaluation to see strategy distribution
        if np.all(final_probs == 0): # Should be prevented by prior normalization and fallback
            chosen_action_idx = 0 # Fold
        else:
            # Ensure final_probs are normalized before choice if any tiny numerical instability
            final_probs = final_probs / np.sum(final_probs)
            chosen_action_idx = np.random.choice(np.arange(self.num_actions), p=final_probs)

        # --- Map action_index to game action string and amount (from PPOPlayer) ---
        # This section relies on PPOPlayer's mapping logic and self.raisesizes.
        # The valid_action_mask check above is critical for ensuring calls to next() are safe.
        chosen_action_str, chosen_amount = "fold", 0
        if chosen_action_idx == 1: # Call/Check
            chosen_action_str = "call"
            # Find the 'call' action in valid_actions; its amount is the amount to call/check.
            call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
            if call_action: chosen_amount = call_action["amount"]
            # If call_action is None (shouldn't happen if mask is correct), it defaults to amount 0.
        elif chosen_action_idx > 1: # Raise
            chosen_action_str = "raise"
            raise_main_action = next((va for va in valid_actions if va['action'] == 'raise'), None)
            if raise_main_action and raise_main_action['amount']['min'] != -1 :
                pot_size = round_state['pot']['main']['amount'] # Simplified pot for raise calc in PPOPlayer
                min_r = raise_main_action['amount']['min']
                max_r = raise_main_action['amount']['max']
                raise_option_idx = chosen_action_idx - 2 # Map network output index to raisesize index
                
                if raise_option_idx < len(self.raisesizes):
                    val = self.raisesizes[raise_option_idx] * pot_size
                    # Clamp to min/max raise amounts allowed by game engine
                    if min_r != -1 and val < min_r: val = min_r
                    if max_r != -1 and val > max_r: val = max_r
                    chosen_amount = int(val)
                else: # Fallback if raise_option_idx is out of bounds for self.raisesizes
                    chosen_amount = min_r if min_r != -1 else 0 # Or some other default like max_r if all-in
                
                # Ensure chosen_amount is valid
                if chosen_amount < min_r and min_r != -1: chosen_amount = min_r
                if chosen_amount > max_r and max_r != -1: chosen_amount = max_r # Should not exceed max_r

                # If the calculated raise amount is invalid (e.g. less than min_raise, or 0 when raise expected)
                if chosen_amount == 0 or (min_r != -1 and chosen_amount < min_r) :
                    if valid_action_mask[1]: # If call/check is valid
                        chosen_action_str = "call"
                        call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
                        if call_action: chosen_amount = call_action["amount"]
                        else: chosen_action_str, chosen_amount = "fold", 0 # Should not happen
                    else: # Fallback to fold if call/check also not valid
                        chosen_action_str = "fold"
                        chosen_amount = 0
            else: # Cannot raise (e.g. min_raise is -1 or raise action not in valid_actions)
                if valid_action_mask[1]: # Try to call/check
                    chosen_action_str = "call"
                    call_action = next((va for va in valid_actions if va['action'] == 'call'), None)
                    if call_action: chosen_amount = call_action["amount"]
                    else: chosen_action_str, chosen_amount = "fold", 0 # Should not happen
                else: # Fallback to fold
                    chosen_action_str = "fold"
                    chosen_amount = 0
        
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
            "chosen_action_idx": chosen_action_idx.item() if isinstance(chosen_action_idx, torch.Tensor) else int(chosen_action_idx), # Ensure it's a Python int
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
        
        log_entry["spot_key"] = get_spot_key(
            round_state['street'], 
            round_state['community_card'], 
            round_state['action_histories']
        )

        self.current_hand_decision_logs.append(log_entry) 
        self.eval_log_list.append(log_entry)

        return chosen_action_str, int(chosen_amount)

    def receive_round_start_message(self, round_count, hole_card, seats):
        super().receive_round_start_message(round_count, hole_card, seats)
        self.current_hand_decision_logs = []

    def receive_round_result_message(self, winners, hand_info, round_state):
        super().receive_round_result_message(winners, hand_info, round_state)

        final_stack = 0
        for player_seat_info in round_state['seats']:
            if player_seat_info['uuid'] == self.uuid:
                final_stack = player_seat_info['stack']
                # self.stack in PPOPlayer is stack _before_ this hand's outcome
                hand_profit_loss = final_stack - self.stack 
                normalized_reward = hand_profit_loss / self.initial_stack # Normalize by game initial stack

                for log_entry in self.current_hand_decision_logs:
                    if log_entry["player_uuid"] == self.uuid:
                        log_entry["hand_reward"] = normalized_reward
                break
        
        self.current_hand_decision_logs = []

# --- Plotting ---
def plot_action_pie_chart(action_freq_dict, spot_key_str, title_prefix_str, output_dir_path):
    """Generates and saves an action frequency pie chart."""
    labels = []
    sizes = []
    colors_list = []
    
    for action_name in Charts.ACTIONS_ORDER:
        if action_name in action_freq_dict and action_freq_dict[action_name] > 0:
            labels.append(f"{action_name.capitalize()} ({action_freq_dict[action_name]})")
            sizes.append(action_freq_dict[action_name])
            colors_list.append(Charts.ACTION_COLORS.get(action_name, '#CCCCCC'))

    if not sizes:
        logger.warning(f"No action frequencies to plot for pie chart (spot: {spot_key_str})")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
    ax.axis('equal')
    ax.set_title(f"{title_prefix_str}: Action Frequencies for Spot: {spot_key_str}")
    
    try:
        # Sanitize spot_key_str for filename if it contains problematic characters
        safe_spot_key_filename = spot_key_str.replace('[','').replace(']','').replace(':','_')
        filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{safe_spot_key_filename}_action_pie.png")
        fig.savefig(filepath)
        logger.info(f"Saved pie chart to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pie chart for spot {spot_key_str} (filename: {safe_spot_key_filename}): {e}")
    finally:
        plt.close(fig)

# --- Main Evaluation Logic ---
def run_evaluation(model_path, num_hands, output_dir, initial_stack_val, small_blind_val, max_rounds_val):
    global evaluation_log
    evaluation_log = []

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Starting evaluation for model: {model_path}, Hands: {num_hands}")
    logger.info(f"Output will be saved to: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- PPO Agent Network Parameters ---
    # IMPORTANT: These parameters MUST match the architecture of the loaded model.
    # They are based on defaults in training.py for PPO.
    # If your trained model used different dimensions, update these accordingly.
    ppo_num_feats = 85
    ppo_num_actions = 11
    ppo_lstm_hidden_size = 128
    ppo_card_embedding_dim = 16
    ppo_num_card_indices_in_state = 7

    actor_critic_net = ActorCriticNetwork(
        input_raw_feat_dim=ppo_num_feats,
        num_actions=ppo_num_actions,
        lstm_hidden_size=ppo_lstm_hidden_size,
        card_embedding_dim=ppo_card_embedding_dim,
        num_card_indices=ppo_num_card_indices_in_state
    ).to(device)

    try:
        actor_critic_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Successfully loaded model weights from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return

    actor_critic_net.eval()

    dummy_optimizer = optim.Adam(actor_critic_net.parameters(), lr=1e-4) 
    dummy_optimizer_path = os.path.join(output_dir, "dummy_ppo_optimizer.pt")

    ppo_eval_player = EvaluationPPOPlayer(
        actor_critic_net=actor_critic_net,
        optimizer=dummy_optimizer,
        model_save_path=model_path,
        optimizer_save_path=dummy_optimizer_path,
        initial_stack=initial_stack_val,
        small_blind=small_blind_val,
        num_feats=ppo_num_feats,
        num_actions=ppo_num_actions,
        lstm_hidden_size=ppo_lstm_hidden_size,
        card_embedding_dim=ppo_card_embedding_dim,
        num_card_indices_in_state=ppo_num_card_indices_in_state,
        eval_log_list=evaluation_log
    )
    
    for i in range(num_hands):
        if (i + 1) % 100 == 0:
            logger.info(f"Simulating hand {i+1}/{num_hands}")

        config = setup_config(max_round=max_rounds_val, initial_stack=initial_stack_val, small_blind_amount=small_blind_val)
        
        players_for_game = [ppo_eval_player]
        for _ in range(5):
            players_for_game.append(HonestPlayer())
        
        random.shuffle(players_for_game)

        for player_idx, player_algo in enumerate(players_for_game):
            config.register_player(name=f"p{player_idx+1}", algorithm=player_algo)
        
        try:
            _ = start_poker(config, verbose=0)
        except Exception as e:
            logger.error(f"Error during game simulation at hand {i+1}: {e}")
            if i > 0 and (i+1) % 10 ==0:
                 logger.info("Continuing simulation despite error in one hand.")

    logger.info(f"Finished {num_hands} simulation hands. Total decisions logged: {len(evaluation_log)}")
    process_evaluation_log(evaluation_log, output_dir, "PPO_Eval")


def process_evaluation_log(log_data, output_dir_path, title_prefix_str):
    if not log_data:
        logger.warning("No evaluation data logged. Skipping processing.")
        return

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

        spot_hand_action_freq_map = {} 
        spot_hand_ev_map = {} 
        spot_action_overall_freq_dict = {action: 0 for action in Charts.ACTIONS_ORDER}

        current_spot_street = decisions_in_spot_list[0]['street'] # All decisions in a spot are on the same street

        for decision in decisions_in_spot_list:
            try:
                hand_type_str = get_hand_type_str_static(decision['hole_cards'])
            except Exception as e:
                logger.error(f"Could not get hand type for {decision['hole_cards']}: {e}")
                hand_type_str = "UnknownHand"

            action_taken = decision['chosen_action_str']
            if action_taken == "call" and decision['chosen_action_amount'] == 0:
                action_taken = "check"

            if action_taken not in spot_action_overall_freq_dict:
                # This case implies an action string not in Charts.ACTIONS_ORDER (raise, call, check, fold)
                # e.g. if chosen_action_str was 'bet'. We should map 'bet' to 'raise' for charts if it's an opening bet.
                # For now, PPOPlayer's output is assumed to be fold, call, raise. Check handles call 0.
                logger.warning(f"Unrecognized action '{action_taken}' in spot {spot_key_str}. It will be ignored for chart aggregation.")
                continue # Skip this decision for aggregation if action is not standard

            if hand_type_str not in spot_hand_action_freq_map:
                spot_hand_action_freq_map[hand_type_str] = {action: 0 for action in Charts.ACTIONS_ORDER}
            if action_taken in spot_hand_action_freq_map[hand_type_str]:
                 spot_hand_action_freq_map[hand_type_str][action_taken] += 1

            if hand_type_str not in spot_hand_ev_map:
                spot_hand_ev_map[hand_type_str] = {'total_ev': 0.0, 'count': 0, 'total_hand_reward': 0.0}
            
            spot_hand_ev_map[hand_type_str]['total_ev'] += decision['estimated_value']
            spot_hand_ev_map[hand_type_str]['total_hand_reward'] += decision['hand_reward']
            spot_hand_ev_map[hand_type_str]['count'] += 1
            
            if action_taken in spot_action_overall_freq_dict:
                 spot_action_overall_freq_dict[action_taken] += 1
        
        # Sanitize spot_key_str for use in filenames to avoid issues with special characters
        # Example: 'FLOP_[C2_D3_H4]_B_R' -> 'FLOP_C2_D3_H4_B_R'
        safe_spot_key_filename = spot_key_str.replace('[','').replace(']','').replace(':','_').replace('/','_')

        # 1. GTO Style Action Grid
        card_action_stat_for_gto_plot = {action: {} for action in Charts.ACTIONS_ORDER}
        for hand_str_key, action_counts_dict in spot_hand_action_freq_map.items():
            for action_name_key, count_val in action_counts_dict.items():
                if count_val > 0:
                    card_action_stat_for_gto_plot[action_name_key][hand_str_key] = count_val
        
        # Pass the actual street name (e.g., 'PREFLOP', 'FLOP') for semantic correctness to the plotting function.
        # The full spot_key_str can be part of the main title or filename.
        street_for_plot_title = current_spot_street.upper() # From the first decision in the spot
        gto_plot_title = f"{title_prefix_str} ({spot_key_str})"

        fig_gto_grid = Charts.plot_gto_style_action_grid(card_action_stat_for_gto_plot, street_for_plot_title, gto_plot_title, player_num=None)
        if fig_gto_grid:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{safe_spot_key_filename}_GTO_action_grid.png")
                fig_gto_grid.savefig(filepath)
                logger.info(f"Saved GTO action grid to {filepath}")
            except Exception as e:
                logger.error(f"Error saving GTO grid for spot {spot_key_str} (filename: {safe_spot_key_filename}): {e}")
            finally:
                plt.close(fig_gto_grid)

        # 2. Action Frequency Pie Chart
        plot_action_pie_chart(spot_action_overall_freq_dict, safe_spot_key_filename, title_prefix_str, output_dir_path)

        # 3. EV Heatmap (critic's estimated value)
        ev_heatmap_data_critic = {
            hand_str_key: data_dict['total_ev'] / data_dict['count'] if data_dict['count'] > 0 else 0
            for hand_str_key, data_dict in spot_hand_ev_map.items() if data_dict['count'] > 0
        }
        critic_ev_plot_title = f"{title_prefix_str} ({spot_key_str}) - Critic EV"
        fig_ev_heatmap_critic = Charts.plot_hand_reward_heatmap(ev_heatmap_data_critic, critic_ev_plot_title, title_prefix_str, player_num=None)
        if fig_ev_heatmap_critic:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{safe_spot_key_filename}_EV_heatmap_critic.png")
                fig_ev_heatmap_critic.savefig(filepath)
                logger.info(f"Saved Critic EV heatmap to {filepath}")
            except Exception as e:
                logger.error(f"Error saving Critic EV heatmap for spot {spot_key_str} (filename: {safe_spot_key_filename}): {e}")
            finally:
                plt.close(fig_ev_heatmap_critic)

        # 3b. EV Heatmap (actual hand reward)
        ev_heatmap_data_actual = {
            hand_str_key: data_dict['total_hand_reward'] / data_dict['count'] if data_dict['count'] > 0 else 0
            for hand_str_key, data_dict in spot_hand_ev_map.items() if data_dict['count'] > 0
        }
        actual_reward_plot_title = f"{title_prefix_str} ({spot_key_str}) - Actual Reward"
        fig_ev_heatmap_actual = Charts.plot_hand_reward_heatmap(ev_heatmap_data_actual, actual_reward_plot_title, title_prefix_str, player_num=None)
        if fig_ev_heatmap_actual:
            try:
                filepath = os.path.join(output_dir_path, f"{title_prefix_str}_{safe_spot_key_filename}_EV_heatmap_actual.png")
                fig_ev_heatmap_actual.savefig(filepath)
                logger.info(f"Saved Actual Reward heatmap to {filepath}")
            except Exception as e:
                logger.error(f"Error saving Actual Reward heatmap for spot {spot_key_str} (filename: {safe_spot_key_filename}): {e}")
            finally:
                plt.close(fig_ev_heatmap_actual)


# --- Argparse and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained PPO Poker Agent.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PPO model (.pt file).")
    parser.add_argument("--num_hands", type=int, default=1000, help="Number of hands to simulate for evaluation.") # Reduced default
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation charts. Defaults to 'evaluation_results/model_name_timestamp'.")
    parser.add_argument("--initial_stack", type=int, default=100, help="Initial chip stack for players.")
    parser.add_argument("--small_blind", type=float, default=0.5, help="Small blind amount.")
    parser.add_argument("--max_rounds", type=int, default=36, help="Max rounds per game (pypokerengine game setting).")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name_no_ext = os.path.splitext(os.path.basename(args.model_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = os.path.join("evaluation_results", f"{model_name_no_ext}_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, "evaluation_log.txt")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    run_evaluation(args.model_path, args.num_hands, args.output_dir, 
                   args.initial_stack, args.small_blind, args.max_rounds)

    logger.info("Evaluation script finished.") 