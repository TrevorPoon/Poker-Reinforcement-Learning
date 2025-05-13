import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

# Standardized action colors
ACTION_COLORS = {
    'raise': '#FF6F61',  # Coral Red
    'call': '#6FA8DC',   # Steel Blue
    'check': '#93C47D',  # Pistachio Green
    'fold': '#FFD966'   # Maize Yellow
}
ACTIONS_ORDER = ['raise', 'call', 'check', 'fold'] # Consistent order

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

def plot_action_proportions(action_stat, TITLE, player_num=None):
    """Generates a figure showing the proportion of actions at each street."""
    # Check if there are any actions recorded
    if not action_stat:
        print(f"Warning: No action stats to plot for player {player_num}")
        return None # Return None if no data
        
    # Prepare data for each street
    streets = list(action_stat.keys())
    # Use the globally defined ACTIONS_ORDER
    
    # Calculate proportions for each action on each street
    proportions = {action: [] for action in ACTIONS_ORDER}
    for street in streets:
        total = sum(action_stat[street].values())  # Total actions on this street
        for action in ACTIONS_ORDER:
            proportion = action_stat[street].get(action, 0) / total if total > 0 else 0
            proportions[action].append(proportion)
    
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = [0] * len(streets)  # Start at zero for each stacked bar

    # Add each action layer to the bar chart using consistent colors and order
    for action in ACTIONS_ORDER:
        ax.bar(streets, proportions[action], label=action, bottom=bottom, color=ACTION_COLORS[action])
        # Update the bottom for the next action layer
        bottom = [i + j for i, j in zip(bottom, proportions[action])]

    # Add labels and title with player number if provided
    ax.set_xlabel("Street")
    ax.set_ylabel("Proportion")
    ax.legend(title="Actions")
    if player_num is not None:
        ax.set_title(f"Proportion of Actions at Each Street (Player {player_num})")
    else:
        ax.set_title("Proportion of Actions at Each Street")
    
    return fig # Return the figure object

def plot_hand_reward_heatmap(hand_reward_stat, TITLE, player_num=None):
    """Generates a heatmap figure of hand rewards."""
    # Check if hand_reward_stat is empty or None
    if not hand_reward_stat:
        print(f"Warning: No hand reward stats to plot for player {player_num}")
        return None # Return None if no data
        
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
    fig, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(reward_matrix, annot=label_matrix, fmt="", cmap="RdYlGn", square=True,
                     linewidths=0.5, cbar_kws={'label': 'Reward Value'},
                     xticklabels=labels, yticklabels=labels)
    
    # Set the plot title and labels with player number if provided
    if player_num is not None:
        ax.set_title(f"Hand Reward Heatmap (Player {player_num})")
    else:
        ax.set_title("Hand Reward Heatmap")

    # Save the heatmap as an image file
    plt.tight_layout()
    return fig # Return the current figure object

def plot_gto_style_action_grid(card_action_stat, street, TITLE, player_num=None):
    """
    Generates a GTO-style action grid figure showing the percentage distribution of actions
    for each hand combination in a grid format for a specific street.
    
    Args:
        card_action_stat (dict): Action frequencies for the specific street.
        street (str): The street name (e.g., 'preflop', 'flop').
        TITLE (str): Title prefix for the generated grid.
        player_num (int, optional): The player number for the chart title.
    """
    # Check if card_action_stat for the specific street is empty or None
    if not card_action_stat:
        print(f"Warning: No card action stats for street '{street}' to plot for player {player_num}")
        return None # Return None if no data
        
    # Define hand labels for the axes
    hand_labels = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    
    # Initialize matrices to store action percentages for each hand
    # Use the globally defined ACTIONS_ORDER and ACTION_COLORS
    num_actions = len(ACTIONS_ORDER)
    grid_data = np.zeros((13, 13, num_actions))  # 13x13 grid with action slots per hand
    
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
            total_count = sum(card_action_stat.get(action, {}).get(hand, 0) for action in ACTIONS_ORDER)
            if total_count > 0:
                for k, action in enumerate(ACTIONS_ORDER):
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
                    action_name = ACTIONS_ORDER[k]
                    ax.add_patch(plt.Rectangle((x, y + bottom), square_size, percentage, color=ACTION_COLORS[action_name]))
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
        plt.Line2D([0], [0], color=ACTION_COLORS['raise'], lw=4, label='Raise'),
        plt.Line2D([0], [0], color=ACTION_COLORS['call'], lw=4, label='Call'),
        plt.Line2D([0], [0], color=ACTION_COLORS['check'], lw=4, label='Check'),
        plt.Line2D([0], [0], color=ACTION_COLORS['fold'], lw=4, label='Fold'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    # Set the title with player number if provided
    title_suffix = f": {street.capitalize()} Action Frequency Grid"
    if player_num is not None:
        ax.set_title(f"{TITLE} (Player {player_num}){title_suffix}", fontsize=16)
    else:
        ax.set_title(f"{TITLE}{title_suffix}", fontsize=16)

    # Save the plot
    plt.tight_layout()
    return fig # Return the figure object

def plot_combined_metric_over_time(metric_data_all_agents, episodes, metric_name, scenario_title):
    """Generates a line plot comparing a single metric across all agents over episodes."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for agent_name, metric_values in metric_data_all_agents.items():
        # Filter out NaN values for plotting, as matplotlib can handle gaps
        # but connecting across large NaN sections might be misleading.
        # We plot what we have.
        valid_episodes = []
        valid_metric_values = []
        for i, val in enumerate(metric_values):
            if not np.isnan(val):
                try:
                    valid_episodes.append(episodes[i])
                    valid_metric_values.append(val)
                except IndexError:
                    # This might happen if metric_values is longer than episodes, though unlikely with current logic
                    print(f"Warning: Mismatch in length for {agent_name}, metric {metric_name}. Skipping some data.")
                    break
        
        if valid_episodes and valid_metric_values: # Ensure there is something to plot
            ax.plot(valid_episodes, valid_metric_values, label=agent_name, marker='o', linestyle='-')

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Combined {metric_name} for All Agents in {scenario_title}")
    ax.legend(loc='best')
    ax.grid(True)
    
    # Apply professional formatting (if desired, and function is available)
    apply_professional_formatting() # Call your existing formatting function

    plt.tight_layout()
    return fig
