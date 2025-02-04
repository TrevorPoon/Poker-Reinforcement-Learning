from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

NB_SIMULATION = 20


class HonestPlayer(BasePokerPlayer):

    def __init__(self):
        self.stat = {'raise': 0, 'call': 0, 'fold': 0}

    def classify_hand(self, card1, card2):
        suits = {'H', 'S', 'D', 'C'}
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

        # Extract rank and suit
        rank1, suit1 = ranks[card1[1]], card1[0]
        rank2, suit2 = ranks[card2[1]], card2[0]
        
        # Check for pairs
        if rank1 == rank2:
            if rank1 >= 8:  # High pairs
                return "Class 1"
            elif 3 <= rank1 <= 7:  # Middle pairs
                return "Class 2"
            else:  # Low pairs
                return "Class 3"
        
        # Check for suited
        suited = suit1 == suit2
        rank_sum = rank1 + rank2
        
        if suited:
            if rank1 >= 13 or rank2 >= 13:  # High suited cards
                return "Class 1"
            elif rank_sum >= 20:  # Medium suited connectors
                return "Class 2"
            else:  # Low suited cards
                return "Class 3"
        else:  # Offsuit hands
            if rank1 == 14 and rank2 == 14:  # High offsuit cards
                return "Class 2"
            elif rank_sum < 30:  # Weak offsuit cards
                return "Class 3"
            else:  # Moderate offsuit cards
                return "Class 2"

    def declare_action(self, valid_actions, hole_card, round_state):
        if self.card_class == "Class 1":
            action = valid_actions[2]
            action['amount'] = action['amount']['min'] # fetch RAISE action info
        elif self.card_class == "Class 2":
            action = valid_actions[1]  # fetch CALL action info
        else:
            if valid_actions[1]['amount'] == 0:
                action = valid_actions[1]
            else: action = valid_actions[0]  # fetch FOLD action info

        # if round_state['street'] == 'preflop':
        #     self.stat[action['action']] += 1
        return action['action'], action['amount']

    def receive_game_start_message(self, game_info):
        self.nb_player = game_info['player_num']

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.card_class = self.classify_hand(hole_card[0], hole_card[1])
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass