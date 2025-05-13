from my_players.DQNPlayer import DQNPlayer
from my_players.NFSPPlayer import NFSPPlayer
from my_players.HonestPlayer_v2 import HonestPlayer
from my_players.cardplayer import cardplayer
from my_players.AllCall import AllCallPlayer
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
plt.style.use('ggplot')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Poker RL Training')
    parser.add_argument('--episodes', type=int, default=10000000, help='Number of episodes')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--scenario', type=str, default='DQN_vs_DQN', 
                        choices=['DQN_vs_Honest', 'DQN_vs_AllCall', 'DQN_vs_DQN',
                                'NFSP_vs_Honest', 'NFSP_vs_AllCall', 'NFSP_vs_NFSP', 'NFSP_vs_DQN'],
                        help='Training scenario')
    parser.add_argument('--training', action='store_true', default=True, help='Training mode')
    parser.add_argument('--agents', type=int, default=None, 
                        help='Number of agents (auto-set based on scenario if not specified)')
    parser.add_argument('--max-rounds', type=int, default=36, help='Max rounds per game')
    parser.add_argument('--initial-stack', type=int, default=100, help='Initial chip stack')
    parser.add_argument('--small-blind', type=float, default=0.5, help='Small blind amount')
    parser.add_argument('--save-interval', type=int, default=1000, help='How often to save metrics to CSV')
    parser.add_argument('--plot-interval', type=int, default=5000, help='How often to generate plots')
    parser.add_argument('--gc-interval', type=int, default=10000, help='How often to run garbage collection')
    return parser.parse_args()

def reset_to_zero(df):
    for key, value in df.items():
        if isinstance(value, dict):  # If value is a dictionary, recurse
            reset_to_zero(value)
        else:  # If value is not a dictionary, reset it to 0
            df[key] = 0
    return df

METRICS_TO_PLOT_COMBINED = ["VPIP", "PFR", "3-Bet", "Model_Loss", "Reward"]

def initialize_agents(scenario, num_agents, training_mode, agent_type=None):
    """Initialize RL agents based on the specified type"""
    agents = []
    
    # Determine agent type from scenario if not explicitly provided
    if agent_type is None:
        agent_type = 'NFSP' if scenario.startswith('NFSP') else 'DQN'
    
    if agent_type == 'DQN':
        for i in range(num_agents):
            model_path = os.getcwd() + f'/models/dqn_agent{i+1}_{scenario}.dump'
            optimizer_path = os.getcwd() + f'/models/dqn_agent{i+1}_optim_{scenario}.dump'
            agents.append(DQNPlayer(model_path, optimizer_path, training_mode))
    elif agent_type == 'NFSP':
        for i in range(num_agents):
            q_model_path = os.getcwd() + f'/models/nfsp_q_agent{i+1}_{scenario}.dump'
            policy_model_path = os.getcwd() + f'/models/nfsp_policy_agent{i+1}_{scenario}.dump'
            q_optimizer_path = os.getcwd() + f'/models/nfsp_q_optim_agent{i+1}_{scenario}.dump'
            policy_optimizer_path = os.getcwd() + f'/models/nfsp_policy_optim_agent{i+1}_{scenario}.dump'
            agents.append(NFSPPlayer(q_model_path, policy_model_path, q_optimizer_path, policy_optimizer_path, training_mode))
    
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
        elif opponent_type in ["DQN", "NFSP"]:
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
               metrics_history, episode_numbers_for_plots, # Added history trackers
               plot_interval=1000, gc_interval=10000):
    """Log metrics using TensorBoard and generate plots periodically."""
    # Only plot periodically
    should_plot = (episode % plot_interval == 0)
    
    if episode % log_interval != 0:
        return True # Return True to indicate training should continue
    
    episode_numbers_for_plots.append(episode) # Log episode number for x-axis of combined plots
    print(f"Episode: {episode+1}")
    
    # No longer need history lists or dataframes here
    loss_switch = 1 # Still needed for potential early stopping
    
    # Detect agent type
    agent_type = 'NFSP' if isinstance(agents[0], NFSPPlayer) else 'DQN'
    
    total_model_loss = 0 # Keep track for early stopping check
    
    for j in range(num_agents):
        agent = agents[j]
        agent_tag_prefix = f"Agent_{j+1}"
        agent_name_for_history = f"Agent_{j+1}" # Consistent key for history dict
        
        # Calculate VPIP, PFR, 3-Bet
        vpip_rate, pfr_rate, three_bet_rate = calculate_stats(agent)
        
        # Append to history for combined plots
        metrics_history["VPIP"][agent_name_for_history].append(vpip_rate)
        metrics_history["PFR"][agent_name_for_history].append(pfr_rate)
        metrics_history["3-Bet"][agent_name_for_history].append(three_bet_rate)
        
        # Log scalar metrics to TensorBoard
        writer.add_scalar(f"{agent_tag_prefix}/VPIP", vpip_rate, episode)
        writer.add_scalar(f"{agent_tag_prefix}/PFR", pfr_rate, episode)
        writer.add_scalar(f"{agent_tag_prefix}/3-Bet", three_bet_rate, episode)
        
        print(f"Agent {j+1} - VPIP: {vpip_rate:.2f}%, PFR: {pfr_rate:.2f}%, 3-Bet: {three_bet_rate:.2f}%")
        
        # Log loss based on agent type
        model_loss_value = np.nan # Default to NaN if no loss
        if agent_type == 'DQN':
            if agent.loss is not None: # Assuming agent.loss is a scalar or can be treated as one
                model_loss_value = agent.loss if isinstance(agent.loss, (int, float)) else np.mean(agent.loss) if isinstance(agent.loss, list) and agent.loss else np.nan
                writer.add_scalar(f"{agent_tag_prefix}/Model_Loss", model_loss_value, episode)
                print(f"Model Loss: {model_loss_value:.5f}")
            else:
                loss_switch = 0
        elif agent_type == 'NFSP':
            if agent.q_loss is not None:
                model_loss_value = agent.q_loss
                writer.add_scalar(f"{agent_tag_prefix}/Q_Network_Loss", model_loss_value, episode)
                print(f"Q-Network Loss: {model_loss_value:.5f}")
            else:
                loss_switch = 0
                # model_loss_value remains np.nan
                
            if agent.policy_loss is not None:
                policy_loss_value = agent.policy_loss
                writer.add_scalar(f"{agent_tag_prefix}/Policy_Network_Loss", policy_loss_value, episode)
                print(f"Policy Network Loss: {policy_loss_value:.5f}")
        
        metrics_history["Model_Loss"][agent_name_for_history].append(model_loss_value)
        total_model_loss += model_loss_value if not np.isnan(model_loss_value) else 0 # Sum valid loss for early stopping check

        # Log reward
        accum_reward = agent.accumulated_reward
        metrics_history["Reward"][agent_name_for_history].append(accum_reward)
        writer.add_scalar(f"{agent_tag_prefix}/Reward", accum_reward, episode)
        print(f"Reward: {accum_reward:.2f}")
        
        # Reset counters for next episode
        agent.VPIP = agent.PFR = agent.three_bet = agent.hand_count = 0
        agent.save_model() # Still save the model state
        
        # Generate and log charts to TensorBoard
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
            print(f"Resetting stats for Agent {j+1} at episode {episode}")

    # Flush writer buffer
    writer.flush()
    
    # Generate and log combined charts to TensorBoard
    if should_plot:
        for metric_name in METRICS_TO_PLOT_COMBINED:
            # Ensure all agents have the same number of data points for this metric as episodes logged
            # This is important if some agents might not report a metric (e.g. loss)
            # For simplicity, we'll assume lengths match episode_numbers_for_plots
            # More robust handling might involve padding or specific checks if lengths can vary significantly
            
            # Create a subset of history for plotting, matching the length of episode_numbers_for_plots
            current_metric_history_for_plot = {}
            for agent_name_key in metrics_history[metric_name]:
                # Take the last N entries, where N is len(episode_numbers_for_plots)
                # This assumes metrics are appended every log_interval when episode_numbers_for_plots is appended
                history_for_agent = metrics_history[metric_name][agent_name_key]
                current_metric_history_for_plot[agent_name_key] = history_for_agent[-len(episode_numbers_for_plots):]


            fig_combined = utils.Charts.plot_combined_metric_over_time(
                current_metric_history_for_plot, 
                episode_numbers_for_plots, 
                metric_name, 
                scenario
            )
            if fig_combined:
                writer.add_figure(f"Combined_Metrics/{metric_name}", fig_combined, episode)
                plt.close(fig_combined)

    # Early stopping check (using average loss across agents)
    # average_loss = total_model_loss / num_agents if num_agents > 0 else 100
    # Check for num_agents being zero and total_model_loss being NaN
    if num_agents > 0:
        valid_losses = [metrics_history["Model_Loss"][f"Agent_{j+1}"][-1] for j in range(num_agents) if metrics_history["Model_Loss"][f"Agent_{j+1}"] and not np.isnan(metrics_history["Model_Loss"][f"Agent_{j+1}"][-1])]
        if valid_losses:
            average_loss = np.mean(valid_losses)
        else:
            average_loss = 100 # Default if no valid losses
    else:
        average_loss = 100

    if loss_switch == 1 and round(average_loss, 5) == 0:
        print(f"Early stopping triggered due to average loss reaching zero.")
        return False # Signal to exit the training loop
        
    # Garbage collection periodically
    if (episode + 1) % gc_interval == 0:
        gc.collect()
        print(f"Garbage collection performed at episode {episode+1}")

    return True # Continue training

def main():
    args = parse_arguments()
    
    # Set parameters from arguments
    num_episodes = args.episodes
    log_interval = args.log_interval
    scenario = args.scenario
    training_mode = args.training
    max_round = args.max_rounds
    initial_stack = args.initial_stack
    small_blind_amount = args.small_blind
    # save_interval = args.save_interval # No longer needed
    plot_interval = args.plot_interval
    gc_interval = args.gc_interval
    
    # Determine agent type from scenario or explicit argument
    agent_type = scenario.split('_vs_')[0]
    
    # Determine number of agents and opponent type
    num_agents = args.agents if args.agents is not None else \
                (6 if scenario.endswith('_vs_NFSP') or scenario.endswith('_vs_DQN') else 1)
    
    opponent_type = scenario.split('_vs_')[1]
    
    # Initialize TensorBoard writer
    log_dir = f"result/runs/{scenario}" # Define log directory
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Initialize metrics history
    metrics_history = {
        metric: {f"Agent_{j+1}": [] for j in range(num_agents)} 
        for metric in METRICS_TO_PLOT_COMBINED
    }
    episode_numbers_for_plots = [] # For X-axis of combined plots

    # Initialize
    # subprocess.run(["python", "src/utils/Clear.py"], check=True) # Optional: Decide if you still need this
    
    # Make sure model directories exist
    os.makedirs('models', exist_ok=True)
    # os.makedirs('result', exist_ok=True) # No longer saving CSV results here
    
    print(f"Starting {agent_type} training against {opponent_type}")
    print(f"Scenario: {scenario}, Episodes: {num_episodes}, Agents: {num_agents}")
    
    # dataframes = initialize_dataframes(scenario) # No longer needed
    agents = initialize_agents(scenario, num_agents, training_mode, agent_type)
    config = setup_game_config(agents, num_agents, opponent_type, max_round, initial_stack, small_blind_amount)
    
    # Training loop
    for i in range(num_episodes):
        game_result = start_poker(config, verbose=0)
        
        # Log metrics and update dataframes
        continue_training = log_metrics(
            agents, num_agents, i, log_interval, scenario, writer, 
            metrics_history, episode_numbers_for_plots, # Pass history trackers
            plot_interval, gc_interval # Pass writer and gc_interval
        )
        
        # Check if we should exit early
        if not continue_training:
            print("Early stopping triggered. Exiting...")
            break
            
        # Garbage collection check moved inside log_metrics

    # Close the TensorBoard writer
    writer.close()
    print("Training finished. TensorBoard writer closed.")

if __name__ == "__main__":
    main()
        

