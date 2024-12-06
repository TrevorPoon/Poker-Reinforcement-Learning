import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    
    # Customizing legend
    plt.legend(loc='upper right', fontsize=8, frameon=True, fancybox=True, framealpha=0.8, shadow=True)
    
    # Light grey background
    plt.gcf().set_facecolor('#f9f9f9')
    plt.gca().set_facecolor('#ffffff')

def plot_line_metric(df, metric_name, ylabel, file_suffix, TITLE, NUM_OF_AGENTS):
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

def plot_action_proportions(action_stat, filename, TITLE):
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

def plot_hand_reward_heatmap(hand_reward_stat, filename, TITLE):
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
