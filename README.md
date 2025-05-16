# Poker Reinforcement Learning

A ML project focused on developing AI agents for playing poker, with primary emphasis on Texas Hold'em.

## Project Description

This project explores the application of RL algorithms to develop strategic poker-playing agents. The system trains AI players that learn optimal poker strategies through self-play and by competing against different opponent types.

The project addresses the complex challenge of decision-making under uncertainty, where agents must learn to:

* Evaluate hand strength across different game stages
* Adapt to opponent strategies
* Make optimal betting decisions based on incomplete information
* Balance exploitative and unexploitable play

The AI agents employ reinforcement learning algorithms including PPO to discover effective poker strategies through extensive training.


## Poker Statistics of PPO Agents in a 6-max 100BB Cash Game

![VPIP of PPO Agents in a 6-max 100BB Cash Game](./images/Poker_Statistics.png)

## Action Tendency of the trained PPO agent in the Preflop Street

![Action Tendency of the trained agent in the Preflop Street](./images/preflop_action_freq.png)

## Installation Instructions

### Prerequisites

* Python 3.11
* CUDA-compatible GPU (recommended for faster training)

### Setup

```bash
git clone https://github.com/TrevorPoon/Poker-Reinforcement-Learning.git
cd Poker-Reinforcement-Learning
mkdir -p models result/runs result/log images
```

## Usage Guide

### Training an Agent

```bash
# Basic training with default parameters (PPO vs PPO)
python src/training.py  

# Training a PPO agent against honest players for 50,000 episodes
python src/training.py --scenario PPO_vs_Honest --episodes 50000  

# Training an NFSP agent with specific game parameters
python src/training.py --scenario NFSP_vs_AllCall --initial-stack 100 --small-blind 0.5 --max-rounds 36
```

### Monitoring Training Progress

```bash
tensorboard --logdir=result/runs
```

This provides visualizations of training metrics including:

* Reward progression
* Model loss
* Poker statistics (VPIP, PFR, 3-Bet)
* Hand reward heatmaps


## Configuration

### training.py arguments:

| Argument        | Description                                  | Default      |
| --------------- | -------------------------------------------- | ------------ |
| --episodes      | Number of training episodes                  | 10,000,000   |
| --log-interval  | Frequency of logging metrics                 | 100          |
| --scenario      | Training scenario                            | PPO\_vs\_PPO |
| --training      | Enable training mode                         | True         |
| --agents        | Number of agents (auto-set if not specified) | None         |
| --max-rounds    | Maximum rounds per poker game                | 36           |
| --initial-stack | Initial chip stack                           | 100          |
| --small-blind   | Small blind amount                           | 0.5          |
| --plot-interval | Frequency of generating visualizations       | 1000         |
| --gc-interval   | Garbage collection frequency                 | 10000        |


## License and Attribution

This project is licensed under the MIT License. See `LICENSE` for more.

Special thanks to **PyPokerEngine** for providing the poker simulation framework.

## Authors

* **Trevor Poon** – Initial work

---






