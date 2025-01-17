from my_players.DQNPlayer import DQNPlayer
from my_players.DQNPlayer1 import DQNPlayer1
from my_players.DQNPlayer2 import DQNPlayer2
from my_players.DQNPlayer3 import DQNPlayer3
from my_players.DQNPlayer4 import DQNPlayer4
from my_players.DQNPlayer5 import DQNPlayer5
from my_players.DQNPlayer6 import DQNPlayer6
from my_players.HonestPlayer import HonestPlayer
from my_players.cardplayer import cardplayer
from my_players.AllCall import AllCallPlayer

import os
import matplotlib.pyplot as plt
import gc
import pandas as pd
import numpy as np
import seaborn as sns
import subprocess
from pypokerengine.api.game import setup_config, start_poker

# User's Input
NUM_EPISODE = 100000000
LOG_INTERVAL = 100
TITLE = 'DQN_vs_AllCall' # 'DQN_vs_AllCall' 'DQN_vs_DQN'
TRAINING = True
NUM_OF_AGENTS = 1


# Initialisation
count = 0
NUM_OF_AGENTS = 6 if TITLE == 'DQN_vs_DQN' else 1
subprocess.run(["python", "src/Utils/Clear.py"], check=True)


def moving_average(data, window_size=10):
    """Calculate the moving average manually."""
    if len(data) < window_size:
        return data  # Not enough data to smooth
    averages = []
    for i in range(len(data) - window_size + 1):
        avg = sum(data[i:i + window_size]) / window_size
        averages.append(avg)
    return averages

# Apply a valid, professional style (like ggplot or seaborn-whitegrid)
plt.style.use('ggplot')

def apply_professional_formatting():
    """Apply professional formatting to the current plot."""
    # Set the font and size for titles, labels, and legend
    plt.title(plt.gca().get_title(), fontsize=16, fontweight='bold', fontname='Arial', color='#333333')
    plt.xlabel(plt.gca().get_xlabel(), fontsize=14, fontweight='bold', fontname='Arial', color='#333333')
    plt.ylabel(plt.gca().get_ylabel(), fontsize=14, fontweight='bold', fontname='Arial', color='#333333')
    
    # Customizing ticks and grid
    plt.xticks(fontsize=12, fontname='Arial', color='#333333')
    plt.yticks(fontsize=12, fontname='Arial', color='#333333')
    
    # Customizing grid
    plt.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    
    # Customizing legend
    plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    
    # Light grey background
    plt.gcf().set_facecolor('#f9f9f9')
    plt.gca().set_facecolor('#ffffff')

def plot_line_metric(df, metric_name, ylabel, file_suffix):
    """Plots both raw and smoothed graphs for a given metric."""
    # Plot raw data
    plt.figure(figsize=(12, 6))
    for j in range(NUM_OF_AGENTS):
        plt.plot(df.iloc[:, j], label=f'Player {j+1} {metric_name}', linewidth=2)
    plt.title(f'{metric_name} History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{TITLE}_{file_suffix}.png')
    plt.close()
    
    # Plot smoothed data
    plt.figure(figsize=(12, 6))
    for j in range(NUM_OF_AGENTS):
        smoothed_data = moving_average(df.iloc[:, j])
        plt.plot(range(len(smoothed_data)), smoothed_data, linestyle='--', 
                 label=f'Player {j+1} Smoothed {metric_name}', alpha=0.8, linewidth=2)
    plt.title(f'Smoothed {metric_name} History of All Agents')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{TITLE}_{file_suffix}_smoothed.png')
    plt.close()

def plot_action_proportions(action_stat, filename):
    # Prepare data for each street
    streets = list(action_stat.keys())
    actions = ['check', 'call', 'raise', 'fold']
    
    # Calculate proportions for each action on each street
    proportions = {action: [] for action in actions}
    for street in streets:
        total = sum(action_stat[street].values())  # Total actions on this street
        for action in actions:
            proportion = action_stat[street][action] / total if total > 0 else 0
            proportions[action].append(proportion)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = [0] * len(streets)  # Start at zero for each stacked bar

    # Add each action layer to the bar chart
    for action in actions:
        ax.bar(streets, proportions[action], label=action, bottom=bottom)
        # Update the bottom for the next action layer
        bottom = [i + j for i, j in zip(bottom, proportions[action])]

    # Add labels and title
    ax.set_xlabel("Street")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportion of Actions at Each Street")
    ax.legend(title="Actions")
    
    # Save the plot as a PNG file
    plt.savefig(f'images/{TITLE}_{filename}', format="png")
    plt.close()

def plot_hand_reward_heatmap(hand_reward_stat, filename):
    # Define hand categories
    hands = hand_reward_stat['hands']
    
    # Initialize a 13x13 matrix for the heatmap data and a label matrix for hand names + values
    reward_matrix = np.full((13, 13), np.nan)
    label_matrix = np.empty((13, 13), dtype=object)
    
    # Define hand labels for the axes
    labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # Populate the reward matrix and label matrix with values from card_reward_stat
    for i, rank1 in enumerate(labels):
        for j, rank2 in enumerate(labels):
            if i == j:  # Pocket pairs
                hand = rank1 + rank2
            elif i < j:  # Suited hands
                hand = rank1 + rank2 + 's'
            else:  # Offsuit hands
                hand = rank2 + rank1 + 'o'
            
            # Assign the reward value to the matrix cell
            reward_value = hands.get(hand, np.nan) / 100
            reward_matrix[i, j] = reward_value
            
            # Create a label with both hand type and reward value, formatted to two decimal places
            label_matrix[i, j] = f"{hand}\n{reward_value:.2f}" if not np.isnan(reward_value) else hand

    # Create the heatmap with both hand labels and values as annotations
    plt.figure(figsize=(12, 12))
    ax = sns.heatmap(reward_matrix, annot=label_matrix, fmt="", cmap="RdYlGn", square=True,
                     linewidths=0.5, cbar_kws={'label': 'Reward Value'},
                     xticklabels=labels, yticklabels=labels)
    
    # Set the plot title and labels
    ax.set_title("Hand Reward Heatmap")

    # Save the heatmap as an image file
    plt.tight_layout()
    plt.savefig(f'images/{TITLE}_{filename}', format="png")
    plt.close()

# Declaration
vpip_history, pfr_history, three_bet_history, loss_history, reward_history = [], [], [], [], []
vpip_df, pfr_df, three_bet_df, loss_df, reward_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data from CSV files if they exist
file_paths = {
    'vpip': f"result/{TITLE}_vpip_history.csv",
    'pfr': f"result/{TITLE}_pfr_history.csv",
    'three_bet': f"result/{TITLE}_three_bet_history.csv",
    'loss': f"result/{TITLE}_loss_history.csv",
    'reward': f"result/{TITLE}_reward_history.csv"
}

dataframes = {
    'vpip': 'vpip_df',
    'pfr': 'pfr_df',
    'three_bet': 'three_bet_df',
    'loss': 'loss_df',
    'reward': 'reward_df'
}

for key, path in file_paths.items():
    if os.path.exists(path):
        try:
            globals()[dataframes[key]] = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Could not load {path}. Error: {e}")

dqn_paths = {}
if NUM_OF_AGENTS == 1:

    dqn_paths = {
        'model': os.getcwd() + f'/models/dqn_{TITLE}.dump',
        'optimizer': os.getcwd() + f'/models/dqn_optim_{TITLE}.dump'
    }

    training_agents =  [DQNPlayer(dqn_paths['model'], dqn_paths['optimizer'], TRAINING)]
else:
    for i in range(NUM_OF_AGENTS):
        dqn_paths[i] = {
            'model': os.getcwd() + f'/models/dqn{i+1}.dump',
            'optimizer': os.getcwd() + f'/models/dqn{i+1}_optim.dump'
        }

    training_agents =  [DQNPlayer1(dqn_paths[0]['model'], dqn_paths[0]['optimizer'], TRAINING),
                        DQNPlayer2(dqn_paths[1]['model'], dqn_paths[1]['optimizer'], TRAINING),
                        DQNPlayer3(dqn_paths[2]['model'], dqn_paths[2]['optimizer'], TRAINING),
                        DQNPlayer4(dqn_paths[3]['model'], dqn_paths[3]['optimizer'], TRAINING),
                        DQNPlayer5(dqn_paths[4]['model'], dqn_paths[4]['optimizer'], TRAINING),
                        DQNPlayer6(dqn_paths[5]['model'], dqn_paths[5]['optimizer'], TRAINING)]

# Set up configuration
config = setup_config(max_round=6, initial_stack=100, small_blind_amount=0.5)

# Register each player with their respective DQNPlayer instance
for i in range(NUM_OF_AGENTS):
    config.register_player(name=f"p{i+1}", algorithm=training_agents[i])

for i in range(NUM_OF_AGENTS+1, 7):
    config.register_player(name=f"p{i+1}", algorithm=AllCallPlayer())

accum_reward = 0

# Game_Simulation
for i in range(0, NUM_EPISODE):
    count += 1
    game_result = start_poker(config, verbose=0)

    if count % LOG_INTERVAL == 0:

        print(count)
        loss_switch = 1
        for j in range(NUM_OF_AGENTS):

            vpip_rate = training_agents[j].VPIP / training_agents[j].hand_count * 100 
            vpip_history.append(vpip_rate)
            print(f"VPIP over each episode: {vpip_rate:.2f}%")

            # Calculate and log PFR
            pfr_rate = training_agents[j].PFR / training_agents[j].hand_count * 100 
            pfr_history.append(pfr_rate)
            print(f"PFR over each episode: {pfr_rate:.2f}%")

            # Calculate and log 3-Bet Percentage
            three_bet_rate = training_agents[j].three_bet / training_agents[j].hand_count * 100
            three_bet_history.append(three_bet_rate)
            print(f"3-Bet Percentage over each episode: {three_bet_rate:.2f}%")

            try:
                model_loss = training_agents[j].loss
                loss_history.append(model_loss)
                print(f"Model Loss: {model_loss:.5f}")
            except:
                loss_switch = 0
            
            accum_reward = training_agents[j].accumulated_reward
            reward_history.append(accum_reward)
            print(f"Reward: {accum_reward:.2f}")

            # Resetting the counts for the next episode
            training_agents[j].VPIP = training_agents[j].PFR = training_agents[j].three_bet = training_agents[j].hand_count = 0
            config.players_info[j]['algorithm'].save_model()

            plot_action_proportions(training_agents[j].action_stat, f"DNQ_Player{j + 1}_action_proportions.png")
            plot_hand_reward_heatmap(training_agents[j].card_reward_stat, f"DNQ_Player{j + 1}_hand_reward_heatmap.png")


        new_vpip = pd.DataFrame([vpip_history])
        new_pfr = pd.DataFrame([pfr_history])
        new_three_bet = pd.DataFrame([three_bet_history])
        new_loss = pd.DataFrame([loss_history])
        new_reward = pd.DataFrame([reward_history])

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

        # Reset index before concatenation
        vpip_df = pd.concat([vpip_df, new_vpip], ignore_index=True)  # No ignore_index here
        pfr_df = pd.concat([pfr_df, new_pfr], ignore_index=True)      # No ignore_index here
        three_bet_df = pd.concat([three_bet_df, new_three_bet],ignore_index=True)
        if loss_switch == 1:
            loss_df = pd.concat([loss_df, new_loss], ignore_index=True)
        reward_df = pd.concat([reward_df, new_reward], ignore_index=True)


        vpip_df.to_csv(f"result/{TITLE}_vpip_history.csv", index=False)
        pfr_df.to_csv(f"result/{TITLE}_pfr_history.csv", index=False)
        three_bet_df.to_csv(f"result/{TITLE}_three_bet_history.csv", index=False)
        loss_df.to_csv(f"result/{TITLE}_loss_history.csv", index=False)
        reward_df.to_csv(f"result/{TITLE}_reward_history.csv", index=False)

        vpip_history, pfr_history, three_bet_history, loss_history, reward_history= [], [], [], [], []

        plot_line_metric(vpip_df, 'VPIP', 'VPIP (%)', 'vpip_history')
        plot_line_metric(pfr_df, 'PFR', 'PFR (%)', 'pfr_history')
        plot_line_metric(three_bet_df, '3-Bet %', '3-Bet %', 'three_bet_history')
        plot_line_metric(reward_df, 'Reward', 'Reward', 'Reward')
        if loss_switch == 1:
            plot_line_metric(loss_df, 'Model Loss', 'Loss', 'model_loss')
            if round(model_loss, 5) == 0:
                break

        if count % 10000 == 0:
            gc.collect()
        

