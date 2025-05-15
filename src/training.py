from my_players.DQNPlayer import DQNPlayer
from my_players.NFSPPlayer import NFSPPlayer
from my_players.HonestPlayer_v2 import HonestPlayer
from my_players.cardplayer import cardplayer
from my_players.AllCall import AllCallPlayer
from my_players.PPOPlayer import PPOPlayer
import utils.Charts 

import os
import gc
import pandas as pd
import numpy as np
import seaborn as sns
import subprocess
import argparse
from pypokerengine.api.game import setup_config, start_poker
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
plt.style.use('ggplot')

# Add for CUDA optimization
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Add for logging
import logging

# Add random for shuffling
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description='Poker RL Training')
    parser.add_argument('--episodes', type=int, default=10000000, help='Number of episodes')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--scenario', type=str, default='DQN_vs_DQN', 
                        choices=['DQN_vs_Honest', 'DQN_vs_AllCall', 'DQN_vs_DQN',
                                'NFSP_vs_Honest', 'NFSP_vs_AllCall', 'NFSP_vs_NFSP',
                                'PPO_vs_Honest', 'PPO_vs_AllCall', 'PPO_vs_PPO'],
                        help='Training scenario')
    parser.add_argument('--training', action='store_true', default=True, help='Training mode')
    parser.add_argument('--agents', type=int, default=None, 
                        help='Number of agents (auto-set based on scenario if not specified)')
    parser.add_argument('--max-rounds', type=int, default=36, help='Max rounds per game')
    parser.add_argument('--initial-stack', type=int, default=100, help='Initial chip stack')
    parser.add_argument('--small-blind', type=float, default=0.5, help='Small blind amount')
    parser.add_argument('--plot-interval', type=int, default=1000, help='How often to generate plots')
    parser.add_argument('--gc-interval', type=int, default=10000, help='How often to run garbage collection')
    return parser.parse_args()

def reset_to_zero(df):
    for key, value in df.items():
        if isinstance(value, dict):  # If value is a dictionary, recurse
            reset_to_zero(value)
        else:  # If value is not a dictionary, reset it to 0
            df[key] = 0
    return df

METRICS_TO_LOG_COMBINED = ["VPIP", "PFR", "3-Bet", "Model_Loss", "Reward"]

def initialize_agents(scenario, num_agents, training_mode, agent_type=None):
    """Initialize RL agents based on the specified type"""
    agents = []
    
    # Determine agent type from scenario if not explicitly provided
    if agent_type is None:
        if scenario.startswith('NFSP'):
            agent_type = 'NFSP'
        elif scenario.startswith('PPO'):
            agent_type = 'PPO'
        else:
            agent_type = 'DQN'
    
    if agent_type == 'DQN':
        for i in range(num_agents):
            model_path = os.getcwd() + f'/models/dqn_agent{i+1}_{scenario}.pt'
            optimizer_path = os.getcwd() + f'/models/dqn_agent{i+1}_optim_{scenario}.pt'
            agents.append(DQNPlayer(model_path, optimizer_path, training_mode))
    elif agent_type == 'NFSP':
        for i in range(num_agents):
            q_model_path = os.getcwd() + f'/models/nfsp_q_agent{i+1}_{scenario}.pt'
            policy_model_path = os.getcwd() + f'/models/nfsp_policy_agent{i+1}_{scenario}.pt'
            q_optimizer_path = os.getcwd() + f'/models/nfsp_q_optim_agent{i+1}_{scenario}.pt'
            policy_optimizer_path = os.getcwd() + f'/models/nfsp_policy_optim_agent{i+1}_{scenario}.pt'
            agents.append(NFSPPlayer(q_model_path, policy_model_path, q_optimizer_path, policy_optimizer_path, training_mode))
    elif agent_type == 'PPO':
        for i in range(num_agents):
            actor_model_path = os.getcwd() + f'/models/ppo_actor_agent{i+1}_{scenario}.pt'
            critic_model_path = os.getcwd() + f'/models/ppo_critic_agent{i+1}_{scenario}.pt'
            actor_optimizer_path = os.getcwd() + f'/models/ppo_actor_optim_agent{i+1}_{scenario}.pt'
            critic_optimizer_path = os.getcwd() + f'/models/ppo_critic_optim_agent{i+1}_{scenario}.pt'
            agents.append(PPOPlayer(actor_model_path, critic_model_path, actor_optimizer_path, critic_optimizer_path, training_mode))
    
    return agents

def setup_game_config(agents, num_agents, opponent_type, max_round, initial_stack, small_blind_amount):
    config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
    
    # Register each player with their respective RL agent instance
    for i in range(num_agents):
        config.register_player(name=f"p{i+1}", algorithm=agents[i])
    
    # Register opponent players
    for i in range(num_agents, 6):
        if opponent_type == "Honest":
            config.register_player(name=f"p{i+1}", algorithm=HonestPlayer())
        elif opponent_type == "AllCall":
            config.register_player(name=f"p{i+1}", algorithm=AllCallPlayer())
        elif opponent_type in ["DQN", "NFSP", "PPO"]:
            # If we're doing self-play, all agents should be registered already
            pass
    
    return config

def calculate_stats(agent):
    # Check for division by zero
    if agent.hand_count == 0:
        return 0, 0, 0 # Return tuple of zeros
        
    # NumPy operations are generally faster than pandas for simple calculations
    vpip_rate = np.divide(agent.VPIP, agent.hand_count) * 100
    pfr_rate = np.divide(agent.PFR, agent.hand_count) * 100
    three_bet_rate = np.divide(agent.three_bet, agent.hand_count) * 100
    
    return vpip_rate, pfr_rate, three_bet_rate

def log_metrics(agents, num_agents, episode, log_interval, scenario, writer, 
               metrics_history, # Still pass metrics_history for other potential uses or detailed data collection
               plot_interval=1000, gc_interval=10000):
    """Log metrics using TensorBoard and generate plots periodically."""
    # Only plot periodically for individual agent charts
    should_plot = (episode % plot_interval == 0)
    
    if episode % log_interval != 0:
        return True # Return True to indicate training should continue
    
        # Generate timestamp for logging

    # episode_numbers_for_plots.append(episode) # No longer needed for this approach
    logger = logging.getLogger(__name__)
    logger.info(f"Episode: {episode+1}")
    
    # No longer need history lists or dataframes here
    loss_switch = 1 # Still needed for potential early stopping
    
    # Detect agent type
    if isinstance(agents[0], NFSPPlayer):
        agent_type = 'NFSP'
    elif isinstance(agents[0], PPOPlayer):
        agent_type = 'PPO'
    else:
        agent_type = 'DQN'
    
    total_model_loss = 0 # Keep track for early stopping check
    
    for j in range(num_agents):
        agent = agents[j]
        agent_tag_prefix = f"Agent_{j+1}"
        agent_name_for_history = f"Agent_{j+1}" # Consistent key for history dict
        
        # Calculate VPIP, PFR, 3-Bet
        vpip_rate, pfr_rate, three_bet_rate = calculate_stats(agent)
        
        # Append to history (can still be useful for other analyses or if user wants raw data later)
        metrics_history["VPIP"][agent_name_for_history].append(vpip_rate)
        metrics_history["PFR"][agent_name_for_history].append(pfr_rate)
        metrics_history["3-Bet"][agent_name_for_history].append(three_bet_rate)
        
        # Log scalar metrics to TensorBoard (individual agent views)
        writer.add_scalar(f"{agent_tag_prefix}/VPIP", vpip_rate, episode)
        writer.add_scalar(f"{agent_tag_prefix}/PFR", pfr_rate, episode)
        writer.add_scalar(f"{agent_tag_prefix}/3-Bet", three_bet_rate, episode)
        
        logger.info(f"Agent {j+1} - VPIP: {vpip_rate:.2f}%, PFR: {pfr_rate:.2f}%, 3-Bet: {three_bet_rate:.2f}%")
        
        # Log loss based on agent type
        model_loss_value = np.nan # Default to NaN if no loss
        
        if agent_type == 'DQN':
            if agent.loss is not None: 
                model_loss_value = agent.loss if isinstance(agent.loss, (int, float)) else np.mean(agent.loss) if isinstance(agent.loss, list) and agent.loss else np.nan
                writer.add_scalar(f"{agent_tag_prefix}/Model_Loss", model_loss_value, episode)
            else:
                loss_switch = 0
        elif agent_type == 'NFSP':
            if agent.q_loss is not None:
                model_loss_value = agent.q_loss # NFSP's primary reported loss for combined metrics
                writer.add_scalar(f"{agent_tag_prefix}/Q_Network_Loss", agent.q_loss, episode)
            else:
                loss_switch = 0
            if agent.policy_loss is not None:
                writer.add_scalar(f"{agent_tag_prefix}/Policy_Network_Loss", agent.policy_loss, episode)
        elif agent_type == 'PPO':
            if agent.loss is not None: # This is actor_loss + critic_loss from PPOPlayer
                model_loss_value = agent.loss
                writer.add_scalar(f"{agent_tag_prefix}/Model_Loss", model_loss_value, episode)
            else:
                loss_switch = 0 # If no combined loss, treat as no loss for early stopping
            
            actor_loss_value = np.nan
            critic_loss_value = np.nan
            if agent.last_loss_actor is not None:
                actor_loss_value = agent.last_loss_actor
                writer.add_scalar(f"{agent_tag_prefix}/Actor_Loss", actor_loss_value, episode)
            if agent.last_loss_critic is not None:
                critic_loss_value = agent.last_loss_critic
                writer.add_scalar(f"{agent_tag_prefix}/Critic_Loss", critic_loss_value, episode)
        
        metrics_history["Model_Loss"][agent_name_for_history].append(model_loss_value)
        total_model_loss += model_loss_value if not np.isnan(model_loss_value) else 0

        # Log reward (individual agent view)
        accum_reward = agent.accumulated_reward
        metrics_history["Reward"][agent_name_for_history].append(accum_reward)
        writer.add_scalar(f"{agent_tag_prefix}/Reward", accum_reward, episode)
        # logger.info(f"Agent {j+1} - Reward: {accum_reward:.2f}")
        
        # Log player stats (individual agent view)
        agent.VPIP = agent.PFR = agent.three_bet = agent.hand_count = 0
        agent.save_model() # Still save the model state
        
        # Generate and log individual agent charts to TensorBoard
        if should_plot:
            # Action Proportions
            fig_actions = utils.Charts.plot_action_proportions(agent.action_stat, scenario, player_num=j+1)
            if fig_actions:
                writer.add_figure(f"{agent_tag_prefix}/Action_Proportions", fig_actions, episode)
                plt.close(fig_actions) # Close the figure after logging

            # Hand Reward Heatmap
            fig_heatmap = utils.Charts.plot_hand_reward_heatmap(agent.card_reward_stat, scenario, player_num=j+1)
            if fig_heatmap:
                writer.add_figure(f"{agent_tag_prefix}/Hand_Reward_Heatmap", fig_heatmap, episode)
                plt.close(fig_heatmap)

            # GTO Style Action Grids for each street
            for street in ['preflop', 'flop', 'turn', 'river']:
                if street in agent.card_action_stat:
                    fig_grid = utils.Charts.plot_gto_style_action_grid(agent.card_action_stat[street], street, scenario, player_num=j+1)
                    if fig_grid:
                        writer.add_figure(f"{agent_tag_prefix}/{street.capitalize()}_Action_Grid", fig_grid, episode)
                        plt.close(fig_grid) # Close the figure
        
        # Reset stats periodically (adjust interval as needed)
        if episode % 10000 == 0 and episode > 0: # Avoid resetting at episode 0
            agent.card_action_stat = reset_to_zero(agent.card_action_stat)
            agent.action_stat = reset_to_zero(agent.action_stat)
            logger.info(f"Resetting stats for Agent {j+1} at episode {episode}")

    # Flush writer buffer
    writer.flush()
    
    # Log combined metrics directly to TensorBoard as scalars for comparison
    for metric_name in METRICS_TO_LOG_COMBINED:
        metric_values = {}
        for k_agent_idx in range(num_agents):
            agent_k_name = f"Agent_{k_agent_idx+1}"
            if metrics_history[metric_name][agent_k_name]:
                val = metrics_history[metric_name][agent_k_name][-1]
                if not np.isnan(val):
                    metric_values[agent_k_name] = val
        
        if metric_values:  # Only log if there's at least one valid entry
            writer.add_scalars(f"Combined_Metrics/{metric_name}", metric_values, episode)

    # Early stopping check (using average loss across agents)
    if num_agents > 0:
        valid_losses = [metrics_history["Model_Loss"][f"Agent_{j+1}"][-1] for j in range(num_agents) if metrics_history["Model_Loss"][f"Agent_{j+1}"] and not np.isnan(metrics_history["Model_Loss"][f"Agent_{j+1}"][-1])]
        if valid_losses:
            average_loss = np.mean(valid_losses)
        else:
            average_loss = 100 # Default if no valid losses
    else:
        average_loss = 100

    if loss_switch == 1 and round(average_loss, 5) == 0:
        logger.info(f"Early stopping triggered due to average loss reaching zero.")
        return False # Signal to exit the training loop
        
    # Garbage collection periodically
    if (episode + 1) % gc_interval == 0:
        gc.collect()
        logger.info(f"Garbage collection performed at episode {episode+1}")

    return True # Continue training

def main():
    args = parse_arguments()
    
    # --- Logger Setup ---
    logger = logging.getLogger(__name__) # Get a logger instance for this module
    logger.setLevel(logging.INFO) # Set the minimum logging level

    log_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Create log directory if it doesn't exist
    log_dir_path = os.path.join(os.getcwd(), "result", "log")
    os.makedirs(log_dir_path, exist_ok=True)
    log_file_path = os.path.join(log_dir_path, f"training_{args.scenario}_{log_timestamp}.log")


    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler() # To also print to console
    stream_handler.setLevel(logging.INFO)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # --- End Logger Setup ---

    # Set parameters from arguments
    num_episodes = args.episodes
    log_interval = args.log_interval
    scenario = args.scenario
    training_mode = args.training
    max_round = args.max_rounds
    initial_stack = args.initial_stack
    small_blind_amount = args.small_blind
    plot_interval = args.plot_interval
    gc_interval = args.gc_interval

    logger.info("--------------------------------")
    logger.info("Arguments received:")
    logger.info(f"Number of episodes: {num_episodes}")
    logger.info(f"Log interval: {log_interval}")
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Training mode: {training_mode}")
    logger.info(f"Max rounds: {max_round}")
    logger.info(f"Initial stack: {initial_stack}")
    logger.info(f"Small blind amount: {small_blind_amount}")
    logger.info(f"Plot interval: {plot_interval}")
    logger.info(f"Garbage collection interval: {gc_interval}")
    logger.info("--------------------------------")
    
    # Determine agent type from scenario or explicit argument
    agent_type = scenario.split('_vs_')[0]
    # Determine number of agents and opponent type
    num_agents = args.agents if args.agents is not None else \
                (6 if scenario.endswith('_vs_NFSP') or scenario.endswith('_vs_DQN') or scenario.endswith('_vs_PPO') else 1)
    opponent_type = scenario.split('_vs_')[1]
    
    # Initialize TensorBoard writer
    tb_log_dir = f"result/runs/{scenario}_{log_timestamp}" # Define log directory for TensorBoard
    writer = SummaryWriter(tb_log_dir)
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    # Initialize metrics history
    metrics_history = {
        metric: {f"Agent_{j+1}": [] for j in range(num_agents)} 
        for metric in METRICS_TO_LOG_COMBINED # Use the renamed list
    }

    # Make sure model directories exist
    os.makedirs('models', exist_ok=True)
    # subprocess.run(['python', 'src/utils/clean.py'], check=True)
    
    logger.info(f"Starting {agent_type} training against {opponent_type}")
    logger.info(f"Scenario: {scenario}, Episodes: {num_episodes}, Agents: {num_agents}")
    
    # dataframes = initialize_dataframes(scenario) # No longer needed
    rl_agents = initialize_agents(scenario, num_agents, training_mode, agent_type)
    
    # Create a list of all player algorithms for shuffling
    all_player_algorithms = list(rl_agents)
    num_total_players = 6 # Assuming a 6-player game
    num_opponents_to_add = num_total_players - num_agents

    if opponent_type == "Honest":
        for _ in range(num_opponents_to_add):
            all_player_algorithms.append(HonestPlayer())
    elif opponent_type == "AllCall":
        for _ in range(num_opponents_to_add):
            all_player_algorithms.append(AllCallPlayer())
    elif opponent_type in ["DQN", "NFSP", "PPO"]:
        # In self-play scenarios, num_agents should be num_total_players,
        # so num_opponents_to_add will be 0.
        # all_player_algorithms already contains all RL agents.
        pass

    # Training loop
    for i in range(num_episodes):
        # Shuffle player positions for the new game
        random.shuffle(all_player_algorithms)
        
        # Setup game configuration for the current episode
        config = setup_config(max_round=max_round, initial_stack=initial_stack, small_blind_amount=small_blind_amount)
        for player_idx, player_algo in enumerate(all_player_algorithms):
            config.register_player(name=f"p{player_idx+1}", algorithm=player_algo)
            # If agents need to know their player_id within the game instance,
            # this might need to be updated if they relied on fixed registration.
            # However, pypokerengine assigns UUIDs and player_id is received in game_start_message.

        game_result = start_poker(config, verbose=0)
        
        # Log metrics and update dataframes
        # When logging, agents are accessed from the rl_agents list, which is not shuffled.
        # Their internal states (like model paths dqn_agent{j+1}) remain consistent.
        # The shuffling only affects their seat in the game.
        continue_training = log_metrics(
            rl_agents, num_agents, i, log_interval, scenario, writer, 
            metrics_history,
            plot_interval, gc_interval
        )
        
        # Check if we should exit early
        if not continue_training:
            logger.info("Early stopping triggered. Exiting...")
            break

    # Close the TensorBoard writer
    writer.close()
    logger.info("Training finished. TensorBoard writer closed.")

if __name__ == "__main__":
    main()
        

