from my_players.DQNPlayer import DQNPlayer
from my_players.DQNPlayer1 import DQNPlayer1   
from my_players.DQNPlayer2 import DQNPlayer2
from my_players.DQNPlayer3 import DQNPlayer3
from my_players.DQNPlayer4 import DQNPlayer4
from my_players.DQNPlayer5 import DQNPlayer5
from my_players.DQNPlayer6 import DQNPlayer6
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
from pypokerengine.api.game import setup_config, start_poker
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# User's Input
NUM_EPISODE = 10000000
LOG_INTERVAL = 100
TITLE = 'DQN_vs_AllCall' # 'DQN_vs_AllCall' 'DQN_vs_DQN' 'DQN_vs_Honest'
TRAINING = True
NUM_OF_AGENTS = 1

# Initialisation
NUM_OF_AGENTS = 6 if TITLE == 'DQN_vs_DQN' else 1
accum_reward = 0
subprocess.run(["python", "src/utils/Clear.py"], check=True)

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
config = setup_config(max_round=36, initial_stack=100, small_blind_amount=0.5)

# Register each player with their respective DQNPlayer instance
for i in range(NUM_OF_AGENTS):
    config.register_player(name=f"p{i+1}", algorithm=training_agents[i])

for i in range(NUM_OF_AGENTS, 6):
    config.register_player(name=f"p{i+1}", algorithm=AllCallPlayer())

def reset_to_zero(df):
    for key, value in df.items():
        if isinstance(value, dict):  # If value is a dictionary, recurse
            reset_to_zero(value)
        else:  # If value is not a dictionary, reset it to 0
            df[key] = 0
    return df 

# Game_Simulation
for i in range(0, NUM_EPISODE):
    game_result = start_poker(config, verbose=0)

    if i % LOG_INTERVAL == 0:
        print(f"Episode: {i+1}")
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

            if training_agents[j].loss:
                model_loss = training_agents[j].loss
                loss_history.append(model_loss)
                print(f"Model Loss: {model_loss:.5f}")
            else:
                loss_switch = 0
                model_loss = 100
            
            accum_reward = training_agents[j].accumulated_reward
            reward_history.append(accum_reward)
            print(f"Reward: {accum_reward:.2f}")

            # Resetting the counts for the next episode
            training_agents[j].VPIP = training_agents[j].PFR = training_agents[j].three_bet = training_agents[j].hand_count = 0
            config.players_info[j]['algorithm'].save_model()

            utils.Charts.plot_action_proportions(training_agents[j].action_stat, f"DNQ_Player{j + 1}_action_proportions.png", TITLE)
            utils.Charts.plot_hand_reward_heatmap(training_agents[j].card_reward_stat, f"DNQ_Player{j + 1}_hand_reward_heatmap.png", TITLE)
            utils.Charts.plot_gto_style_action_grid(training_agents[j].card_action_stat['preflop'], f"DNQ_Player{j + 1}_preflop_action_grid.png", TITLE)
            utils.Charts.plot_gto_style_action_grid(training_agents[j].card_action_stat['flop'], f"DNQ_Player{j + 1}_flop_action_grid.png", TITLE)
            utils.Charts.plot_gto_style_action_grid(training_agents[j].card_action_stat['river'], f"DNQ_Player{j + 1}_river_action_grid.png", TITLE)
            utils.Charts.plot_gto_style_action_grid(training_agents[j].card_action_stat['turn'], f"DNQ_Player{j + 1}_turn_action_grid.png", TITLE)


            if i % 10000 == 0: 
                card_action_stat = reset_to_zero(training_agents[j].card_action_stat)
                action_stat = reset_to_zero(training_agents[j].action_stat)

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

        utils.Charts.plot_line_metric(vpip_df, 'VPIP', 'VPIP (%)', 'vpip_history', TITLE, NUM_OF_AGENTS)
        utils.Charts.plot_line_metric(pfr_df, 'PFR', 'PFR (%)', 'pfr_history', TITLE, NUM_OF_AGENTS)
        utils.Charts.plot_line_metric(three_bet_df, '3-Bet %', '3-Bet %', 'three_bet_history', TITLE, NUM_OF_AGENTS)
        utils.Charts.plot_line_metric(reward_df, 'Reward', 'Reward', 'Reward', TITLE, NUM_OF_AGENTS)

        if loss_switch == 1:
            utils.Charts.plot_line_metric(loss_df, 'Model Loss', 'Loss', 'model_loss', TITLE, NUM_OF_AGENTS)
            if round(model_loss, 5) == 0: # Early Stopping
                exit()

        if i + 1 % 10000 == 0:
            gc.collect()
        

