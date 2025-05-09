import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

def moving_average(data, window_size=10):
    """Calculate the moving average manually."""
    if len(data) < window_size:
        return data  # Not enough data to smooth
    averages = []
    for i in range(len(data) - window_size + 1):
        avg = sum(data[i:i + window_size]) / window_size
        averages.append(avg)
    return averages

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
    
    # Only add legend if there are labeled artists
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles and labels:
        plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    
    # Light grey background
    plt.gcf().set_facecolor('#f9f9f9')
    plt.gca().set_facecolor('#ffffff')

def plot_line_metric(df, metric_name, ylabel, file_suffix, TITLE, NUM_OF_AGENTS):
    """Plots both raw and smoothed graphs for a given metric."""
    # Check if dataframe is empty
    if df.empty:
        print(f"Warning: Empty dataframe, cannot plot {metric_name}")
        return
        
    # Plot raw data
    plt.figure(figsize=(12, 6))
    # Only plot data for agents that have columns in the dataframe
    actual_columns = min(NUM_OF_AGENTS, df.shape[1])
    for j in range(actual_columns):
        plt.plot(df.iloc[:, j], label=f'Player {j+1} {metric_name}', linewidth=2)
    plt.title(f'{metric_name} History of All Players (Total: {NUM_OF_AGENTS})')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{TITLE}_{file_suffix}_players.png')
    plt.close()
    
    # Plot smoothed data
    plt.figure(figsize=(12, 6))
    for j in range(actual_columns):
        # Skip if series is empty
        if len(df.iloc[:, j]) == 0:
            continue
        smoothed_data = moving_average(df.iloc[:, j])
        plt.plot(range(len(smoothed_data)), smoothed_data, linestyle='--', 
                 label=f'Player {j+1} Smoothed {metric_name}', alpha=0.8, linewidth=2)
    plt.title(f'Smoothed {metric_name} History of All Players (Total: {NUM_OF_AGENTS})')
    plt.xlabel('Episodes (every 100 episodes)')
    plt.ylabel(ylabel)
    apply_professional_formatting()
    plt.savefig(f'images/{TITLE}_{file_suffix}_smoothed_players.png')
    plt.close()

def plot_action_proportions(action_stat, filename, TITLE, player_num=None):
    # Check if there are any actions recorded
    if not action_stat:
        print(f"Warning: No action stats to plot for {filename}")
        return
        
    # Prepare data for each street
    streets = list(action_stat.keys())
    actions = ['check', 'call', 'raise', 'fold']
    
    # Calculate proportions for each action on each street
    proportions = {action: [] for action in actions}
    for street in streets:
        total = sum(action_stat[street].values())  # Total actions on this street
        for action in actions:
            proportion = action_stat[street].get(action, 0) / total if total > 0 else 0
            proportions[action].append(proportion)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = [0] * len(streets)  # Start at zero for each stacked bar

    # Add each action layer to the bar chart
    for action in actions:
        ax.bar(streets, proportions[action], label=action, bottom=bottom)
        # Update the bottom for the next action layer
        bottom = [i + j for i, j in zip(bottom, proportions[action])]

    # Add labels and title with player number if provided
    ax.set_xlabel("Street")
    ax.set_ylabel("Proportion")
    if player_num is not None:
        ax.set_title(f"Proportion of Actions at Each Street (Player {player_num})")
        plt.savefig(f'images/{TITLE}_{filename}_player_{player_num}.png', format="png")
    else:
        ax.set_title("Proportion of Actions at Each Street")
        plt.savefig(f'images/{TITLE}_{filename}.png', format="png")
    
    ax.legend(title="Actions")
    plt.close()

def plot_hand_reward_heatmap(hand_reward_stat, filename, TITLE, player_num=None):
    # Check if hand_reward_stat is empty or None
    if not hand_reward_stat:
        print(f"Warning: No hand reward stats to plot for {filename}")
        return
        
    # Define hand categories
    hands = hand_reward_stat
    
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
    
    # Set the plot title and labels with player number if provided
    if player_num is not None:
        ax.set_title(f"Hand Reward Heatmap (Player {player_num})")
        plt.savefig(f'images/{TITLE}_{filename}_player_{player_num}.png', format="png")
    else:
        ax.set_title("Hand Reward Heatmap")
        plt.savefig(f'images/{TITLE}_{filename}.png', format="png")

    # Save the heatmap as an image file
    plt.tight_layout()
    plt.close()

def plot_gto_style_action_grid(card_action_stat, filename, TITLE, player_num=None):
    """
    Plot a GTO-style action grid showing the percentage distribution of actions
    for each hand combination in a grid format.
    
    Args:
        card_action_stat (dict): Nested dictionary with actions and hand frequencies.
        filename (str): Name of the output image file.
        TITLE (str): Title for the generated grid.
        player_num (int, optional): The player number for the chart title.
    """
    # Check if card_action_stat is empty or None
    if not card_action_stat:
        print(f"Warning: No card action stats to plot for {filename}")
        return
        
    # Define hand labels for the axes
    hand_labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # Initialize matrices to store action percentages for each hand
    action_colors = ['#FF6F61', '#6FA8DC', '#93C47D', '#FFD966']  # Colors for raise, call, check, fold
    actions = ['raise', 'call', 'check', 'fold']
    num_actions = len(actions)
    grid_data = np.zeros((13, 13, num_actions))  # 13x13 grid with 4 action slots per hand
    
    # Populate the grid with action frequencies
    for i, rank1 in enumerate(hand_labels):
        for j, rank2 in enumerate(hand_labels):
            if i == j:  # Pocket pairs
                hand = rank1 + rank2
            elif i < j:  # Suited hands
                hand = rank1 + rank2 + 's'
            else:  # Offsuit hands
                hand = rank2 + rank1 + 'o'

            # Retrieve action counts for the hand
            total_count = sum(card_action_stat.get(action, {}).get(hand, 0) for action in actions)
            if total_count > 0:
                for k, action in enumerate(actions):
                    grid_data[i, j, k] = card_action_stat.get(action, {}).get(hand, 0) / total_count  # Normalize

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')

    # Draw squares with action percentages
    for i in range(13):
        for j in range(13):
            # Extract the action percentages for the current hand
            percentages = grid_data[i, j]
            if percentages.sum() == 0:
                continue  # Skip empty squares

            # Define the square boundaries
            x, y = j, 12 - i  # Flip the y-axis for AA at top-left and 22 at bottom-right
            square_size = 1
            bottom = 0

            # Draw segments for each action in the square
            for k, percentage in enumerate(percentages):
                if percentage > 0:
                    ax.add_patch(plt.Rectangle((x, y + bottom), square_size, percentage, color=action_colors[k]))
                    bottom += percentage

            # Add text label in the center
            hand_label = hand_labels[i] + hand_labels[j] if i == j else (
                hand_labels[i] + hand_labels[j] + 's' if i < j else hand_labels[j] + hand_labels[i] + 'o')
            ax.text(x + 0.5, y + 0.5, hand_label, ha='center', va='center', fontsize=8, color='black')

    # Set ticks and labels
    ax.set_xticks(np.arange(13) + 0.5)
    ax.set_yticks(np.arange(13) + 0.5)
    ax.set_xticklabels(hand_labels)
    ax.set_yticklabels(hand_labels[::-1])  # Reverse y-axis labels

    # Add gridlines to separate squares
    ax.hlines(y=np.arange(14), xmin=0, xmax=13, color='black', linewidth=1)
    ax.vlines(x=np.arange(14), ymin=0, ymax=13, color='black', linewidth=1)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=action_colors[0], lw=4, label='Raise'),
        plt.Line2D([0], [0], color=action_colors[1], lw=4, label='Call'),
        plt.Line2D([0], [0], color=action_colors[2], lw=4, label='Check'),
        plt.Line2D([0], [0], color=action_colors[3], lw=4, label='Fold'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    # Set the title with player number if provided
    if player_num is not None:
        ax.set_title(f"{TITLE}: Action Frequency Grid (Player {player_num})", fontsize=16)
        plt.savefig(f'images/{TITLE}_{filename}_player_{player_num}.png', format="png")
    else:
        ax.set_title(f"{TITLE}: Action Frequency Grid", fontsize=16)
        plt.savefig(f'images/{TITLE}_{filename}.png', format="png")

    # Save the plot
    plt.tight_layout()
    plt.close()
