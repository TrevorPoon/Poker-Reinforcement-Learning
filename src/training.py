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

def initialize_dataframes(scenario):
    # Load data from CSV files if they exist
    file_paths = {
        'vpip': f"result/{scenario}_vpip_history.csv",
        'pfr': f"result/{scenario}_pfr_history.csv",
        'three_bet': f"result/{scenario}_three_bet_history.csv",
        'loss': f"result/{scenario}_loss_history.csv",
        'reward': f"result/{scenario}_reward_history.csv",
        'policy_loss': f"result/{scenario}_policy_loss_history.csv",  # For NFSP
        'q_loss': f"result/{scenario}_q_loss_history.csv"  # For NFSP
    }
    
    dataframes = {
        'vpip_df': pd.DataFrame(),
        'pfr_df': pd.DataFrame(),
        'three_bet_df': pd.DataFrame(),
        'loss_df': pd.DataFrame(),
        'reward_df': pd.DataFrame(),
        'policy_loss_df': pd.DataFrame(),  # For NFSP
        'q_loss_df': pd.DataFrame()  # For NFSP
    }
    
    for key, path in file_paths.items():
        df_name = f"{key}_df"
        if os.path.exists(path):
            try:
                dataframes[df_name] = pd.read_csv(path)
            except Exception as e:
                print(f"Warning: Could not load {path}. Error: {e}")
    
    return dataframes

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
        return 0, 0, 0
        
    # NumPy operations are generally faster than pandas for simple calculations
    vpip_rate = np.divide(agent.VPIP, agent.hand_count) * 100
    pfr_rate = np.divide(agent.PFR, agent.hand_count) * 100
    three_bet_rate = np.divide(agent.three_bet, agent.hand_count) * 100
    
    return vpip_rate, pfr_rate, three_bet_rate

def log_metrics(agents, num_agents, episode, log_interval, dataframes, scenario, 
               save_interval=1000, plot_interval=5000):
    """Log and save metrics for both DQN and NFSP agents"""
    # Only save CSV files periodically instead of every log interval
    should_save_csv = (episode % save_interval == 0)
    should_plot = (episode % plot_interval == 0)
    
    if episode % log_interval != 0:
        return dataframes
    
    print(f"Episode: {episode+1}")
    
    vpip_history, pfr_history, three_bet_history = [], [], []
    loss_history, reward_history = [], []
    policy_loss_history, q_loss_history = [], []  # For NFSP
    loss_switch = 1
    
    # Detect agent type
    agent_type = 'NFSP' if isinstance(agents[0], NFSPPlayer) else 'DQN'
    
    for j in range(num_agents):
        agent = agents[j]
        
        # Calculate and log VPIP, PFR, 3-Bet
        vpip_rate, pfr_rate, three_bet_rate = calculate_stats(agent)
        vpip_history.append(vpip_rate)
        pfr_history.append(pfr_rate)
        three_bet_history.append(three_bet_rate)
        
        print(f"Agent {j+1} - VPIP: {vpip_rate:.2f}%, PFR: {pfr_rate:.2f}%, 3-Bet: {three_bet_rate:.2f}%")
        
        # Log loss based on agent type
        if agent_type == 'DQN':
            if agent.loss:
                model_loss = agent.loss
                loss_history.append(model_loss)
                print(f"Model Loss: {model_loss:.5f}")
            else:
                loss_switch = 0
                model_loss = 100
        elif agent_type == 'NFSP':
            # For NFSP, record both Q-network and policy network losses
            if agent.q_loss is not None:
                q_loss_history.append(agent.q_loss)
                print(f"Q-Network Loss: {agent.q_loss:.5f}")
                model_loss = agent.q_loss  # Use Q-loss for early stopping check
            else:
                q_loss_history.append(0)
                model_loss = 100
                loss_switch = 0
                
            if agent.policy_loss is not None:
                policy_loss_history.append(agent.policy_loss)
                print(f"Policy Network Loss: {agent.policy_loss:.5f}")
            else:
                policy_loss_history.append(0)
        
        # Log reward
        accum_reward = agent.accumulated_reward
        reward_history.append(accum_reward)
        print(f"Reward: {accum_reward:.2f}")
        
        # Reset counters for next episode
        agent.VPIP = agent.PFR = agent.three_bet = agent.hand_count = 0
        agent.save_model()
        
        # Generate charts
        if should_plot:
            agent_prefix = "NFSP" if agent_type == "NFSP" else "DQN"
            utils.Charts.plot_action_proportions(agent.action_stat, f"Player{j+1}_action_proportions.png", scenario)
            utils.Charts.plot_hand_reward_heatmap(agent.card_reward_stat, f"Player{j+1}_hand_reward_heatmap.png", scenario)
            utils.Charts.plot_gto_style_action_grid(agent.card_action_stat['preflop'], f"Player{j+1}_preflop_action_grid.png", scenario)
            utils.Charts.plot_gto_style_action_grid(agent.card_action_stat['flop'], f"Player{j+1}_flop_action_grid.png", scenario)
            utils.Charts.plot_gto_style_action_grid(agent.card_action_stat['river'], f"Player{j+1}_river_action_grid.png", scenario)
            utils.Charts.plot_gto_style_action_grid(agent.card_action_stat['turn'], f"Player{j+1}_turn_action_grid.png", scenario)
        
        # Reset stats periodically
        if episode % 10000 == 0:
            agent.card_action_stat = reset_to_zero(agent.card_action_stat)
            agent.action_stat = reset_to_zero(agent.action_stat)
    
    # Update dataframes with new metrics
    vpip_df = dataframes['vpip_df']
    pfr_df = dataframes['pfr_df']
    three_bet_df = dataframes['three_bet_df']
    loss_df = dataframes['loss_df']
    reward_df = dataframes['reward_df']
    
    # NFSP-specific dataframes
    policy_loss_df = dataframes['policy_loss_df'] if agent_type == 'NFSP' else None
    q_loss_df = dataframes['q_loss_df'] if agent_type == 'NFSP' else None
    
    # Create new dataframes for this iteration
    new_vpip = pd.DataFrame([vpip_history])
    new_pfr = pd.DataFrame([pfr_history])
    new_three_bet = pd.DataFrame([three_bet_history])
    new_loss = pd.DataFrame([loss_history])
    new_reward = pd.DataFrame([reward_history])
    
    # NFSP-specific new dataframes
    if agent_type == 'NFSP':
        new_policy_loss = pd.DataFrame([policy_loss_history])
        new_q_loss = pd.DataFrame([q_loss_history])
    
    # Ensure column alignment for concatenation
    if not vpip_df.empty:
        new_vpip.columns = vpip_df.columns
    if not pfr_df.empty:
        new_pfr.columns = pfr_df.columns
    if not three_bet_df.empty:
        new_three_bet.columns = three_bet_df.columns
    if not loss_df.empty:
        new_loss.columns = loss_df.columns
    if not reward_df.empty:
        new_reward.columns = reward_df.columns
        
    # NFSP-specific column alignment
    if agent_type == 'NFSP':
        if not policy_loss_df.empty:
            new_policy_loss.columns = policy_loss_df.columns
        if not q_loss_df.empty:
            new_q_loss.columns = q_loss_df.columns
    
    # Concatenate new data
    vpip_df = pd.concat([vpip_df, new_vpip], ignore_index=True)
    pfr_df = pd.concat([pfr_df, new_pfr], ignore_index=True)
    three_bet_df = pd.concat([three_bet_df, new_three_bet], ignore_index=True)
    if loss_switch == 1:
        loss_df = pd.concat([loss_df, new_loss], ignore_index=True)
    reward_df = pd.concat([reward_df, new_reward], ignore_index=True)
    
    # NFSP-specific concatenation
    if agent_type == 'NFSP':
        policy_loss_df = pd.concat([policy_loss_df, new_policy_loss], ignore_index=True)
        q_loss_df = pd.concat([q_loss_df, new_q_loss], ignore_index=True)
    
    # Save to CSV conditionally
    if should_save_csv:
        vpip_df.to_csv(f"result/{scenario}_vpip_history.csv", index=False)
        pfr_df.to_csv(f"result/{scenario}_pfr_history.csv", index=False)
        three_bet_df.to_csv(f"result/{scenario}_three_bet_history.csv", index=False)
        loss_df.to_csv(f"result/{scenario}_loss_history.csv", index=False)
        reward_df.to_csv(f"result/{scenario}_reward_history.csv", index=False)
        
        # NFSP-specific CSV saving
        if agent_type == 'NFSP':
            policy_loss_df.to_csv(f"result/{scenario}_policy_loss_history.csv", index=False)
            q_loss_df.to_csv(f"result/{scenario}_q_loss_history.csv", index=False)
    
    # Plot metrics
    if should_plot:
        # Create the images directory if it doesn't exist
        os.makedirs('images', exist_ok=True)
        
        # Common metrics that apply to all agent types
        utils.Charts.plot_line_metric(vpip_df, 'VPIP', 'VPIP (%)', 'vpip_history', scenario, num_agents)
        utils.Charts.plot_line_metric(pfr_df, 'PFR', 'PFR (%)', 'pfr_history', scenario, num_agents)
        utils.Charts.plot_line_metric(three_bet_df, '3-Bet %', '3-Bet %', 'three_bet_history', scenario, num_agents)
        utils.Charts.plot_line_metric(reward_df, 'Reward', 'Reward', 'Reward', scenario, num_agents)
        
        # Agent-specific metrics
        if agent_type == 'DQN' and loss_switch == 1 and not loss_df.empty:
            utils.Charts.plot_line_metric(loss_df, 'Model Loss', 'Loss', 'model_loss', scenario, num_agents)
        elif agent_type == 'NFSP':
            if not policy_loss_df.empty and policy_loss_df.shape[1] > 0:
                utils.Charts.plot_line_metric(policy_loss_df, 'Policy Loss', 'Loss', 'policy_loss', scenario, num_agents)
            if not q_loss_df.empty and q_loss_df.shape[1] > 0:
                utils.Charts.plot_line_metric(q_loss_df, 'Q-Network Loss', 'Loss', 'q_loss', scenario, num_agents)
    
    # Early stopping check
    if loss_switch == 1 and round(model_loss, 5) == 0:
        return None  # Signal to exit the training loop
    
    # Update dataframes dictionary
    dataframes['vpip_df'] = vpip_df
    dataframes['pfr_df'] = pfr_df
    dataframes['three_bet_df'] = three_bet_df
    dataframes['loss_df'] = loss_df
    dataframes['reward_df'] = reward_df
    
    # NFSP-specific dictionary updates
    if agent_type == 'NFSP':
        dataframes['policy_loss_df'] = policy_loss_df
        dataframes['q_loss_df'] = q_loss_df
    
    return dataframes

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
    save_interval = args.save_interval
    plot_interval = args.plot_interval
    gc_interval = args.gc_interval
    
    # Determine agent type from scenario or explicit argument
    agent_type = scenario.split('_vs_')[0]
    
    # Determine number of agents and opponent type
    num_agents = args.agents if args.agents is not None else \
                (6 if scenario.endswith('_vs_NFSP') or scenario.endswith('_vs_DQN') else 1)
    
    opponent_type = scenario.split('_vs_')[1]
    
    # Initialize
    subprocess.run(["python", "src/utils/Clear.py"], check=True)
    
    # Make sure model directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('result', exist_ok=True)
    
    print(f"Starting {agent_type} training against {opponent_type}")
    print(f"Scenario: {scenario}, Episodes: {num_episodes}, Agents: {num_agents}")
    
    dataframes = initialize_dataframes(scenario)
    agents = initialize_agents(scenario, num_agents, training_mode, agent_type)
    config = setup_game_config(agents, num_agents, opponent_type, max_round, initial_stack, small_blind_amount)
    
    # Training loop
    for i in range(num_episodes):
        game_result = start_poker(config, verbose=0)
        
        # Log metrics and update dataframes
        dataframes = log_metrics(
            agents, num_agents, i, log_interval, dataframes, scenario, 
            save_interval, plot_interval
        )
        
        # Check if we should exit early
        if dataframes is None:
            print("Early stopping triggered. Exiting...")
            break
            
        # Garbage collection periodically
        if (i + 1) % gc_interval == 0:
            gc.collect()
            print(f"Garbage collection performed at episode {i+1}")

if __name__ == "__main__":
    main()
        

