import os.path
from players.base_player import Player
import re
import json
from games.base_game import Game
from utils.utils import pretty_number


class BargainingGame(Game):
    def __init__(self, player_1: Player, player_2: Player, data_logger, money_to_divide, max_rounds,
                 complete_information, messages_allowed, show_inflation_update=True):
        super().__init__(player_1, player_2, data_logger)
        self.delta_1 = player_1.delta
        self.delta_2 = player_2.delta
        self.max_rounds = max_rounds
        self.delta_1_loss = (1 - self.delta_1) * 100
        self.delta_2_loss = (1 - self.delta_2) * 100
        self.money_to_divide = money_to_divide
        self.complete_information = complete_information
        self.messages_allowed = messages_allowed
        self.show_inflation_update = show_inflation_update

        assert player_1.public_name != player_2.public_name, "Players should have different names"

        self.player_1.new_chat()
        self.player_2.new_chat()

        p1_info, p1_rules = self.construct_player_info('p1')
        p2_info, p2_rules = self.construct_player_info('p2')

        self.all_names = [self.player_1.public_name, self.player_2.public_name]

        path = 'games/bargaining/prompts/'
        self.set_rules(os.path.join(path, 'rules_prompt.txt'), p1_info, 'p1')
        self.set_rules(os.path.join(path, 'rules_prompt.txt'), p2_info, 'p2')

        self.add_format(os.path.join(path, 'offer_format.txt'), p1_rules, 'p1')
        self.add_format(os.path.join(path, 'offer_format.txt'), p2_rules, 'p2')
        self.game()

    def build_max_round(self):
        if 0 < self.max_rounds <= 20:
            message = f"You have {self.max_rounds} rounds to divide the money, or both of you will get nothing!\n"
        else:
            message = ""
        return message

    def build_inflation_message(self, player='p1'):
        self_loss = self.delta_1_loss if player == "p1" else self.delta_2_loss
        other_loss = self.delta_2_loss if player == "p1" else self.delta_1_loss
        other_name = self.player_2.public_name if player == "p1" else self.player_1.public_name
        if not self_loss and not other_loss:  # No inflation
            return ""
        prefix = f"Beware of inflation! With each passing round, the money is worth {pretty_number(self_loss)}% less "
        if self.complete_information:
            if self.delta_1_loss != self.delta_2_loss:
                prefix += f'for you. For {other_name}, the money is worth {pretty_number(other_loss)}% less'
        else:
            prefix += f"for you. You don't know how the inflation effect on {other_name}"
        return prefix.strip() + '.'

    def get_other_player(self, base='p1'):
        base_player = self.player_1 if base == 'p1' else self.player_2
        other_player = self.player_2 if base == 'p1' else self.player_1
        return base_player, other_player

    def construct_player_info(self, player='p1'):
        base_player, other_player = self.get_other_player(player)
        player_info = dict()
        player_info['{{player_name}}'] = base_player.public_name
        player_info['{{rival_name}}'] = other_player.public_name
        player_info['{{inflation_message}}'] = self.build_inflation_message(player)
        player_info['{{max_rounds_message}}'] = self.build_max_round()
        player_info['{{money_to_divide}}'] = pretty_number(self.money_to_divide)
        player_info['{{player_1_public_name}}'] = self.player_1.public_name
        player_info['{{player_2_public_name}}'] = self.player_2.public_name
        player_info['{{delta_1_loss}}'] = pretty_number(self.delta_1_loss)
        player_info['{{delta_2_loss}}'] = pretty_number(self.delta_2_loss)

        add_message_str1 = " and the message you attached"
        add_message_str2 = f',\n"message": The message you pass to {other_player.public_name}'

        rules_info = dict()
        rules_info['{{money_to_divide}}'] = pretty_number(self.money_to_divide)
        rules_info['{{player_name_lower}}'] = self.lower_name(base_player.public_name)
        rules_info['{{rival_name_lower}}'] = self.lower_name(other_player.public_name)
        rules_info['{{rival_name}}'] = other_player.public_name
        rules_info['{{add_message1}}'] = add_message_str1 if self.messages_allowed else ""
        rules_info['{{add_message2}}'] = add_message_str2 if self.messages_allowed else ""
        return player_info, rules_info

    @staticmethod
    def lower_name(name):
        return name.lower().replace(" ", "_")

    def is_offer_in_format(self, string_action):
        # clean the string from words that are not inside the JSON format
        action = re.search(r'\{.*?\}', string_action, re.DOTALL)
        action = action.group() if action else ""
        action = re.sub(r"(?<=\d),(?=\d{3})", "", action)

        # Then, check if JSON is in format
        try:
            action = json.loads(action)
            if not isinstance(action, dict):
                return False
            if not all(key in action for key in
                       [f"{self.lower_name(name)}_gain" for name in self.all_names] +
                       (["message"] if self.messages_allowed else [])):
                return False
            if not all(isinstance(action[key], (int, float)) for key in
                       [f"{self.lower_name(name)}_gain" for name in self.all_names]):
                return False
            total_gain = [action[someone_gain] for someone_gain in action.keys() if "_gain" in someone_gain]
            if sum(total_gain) != self.money_to_divide:
                return False
            if self.messages_allowed and not isinstance(action["message"], str):
                return False
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def is_decision_in_format(string_action):
        # clean the string from words that are not inside the JSON format
        action = re.search(r'\{.*?\}', string_action, re.DOTALL)
        action = action.group() if action else ""
        # Then, check if JSON is in format
        try:
            action = json.loads(action)
            if not isinstance(action, dict):
                return False
            if not all(key in action for key in
                       ["decision"]):
                return False
            if not isinstance(action["decision"], str):
                return False
            if action["decision"] not in ["accept", "reject"]:
                return False
            return True
        except json.JSONDecodeError:
            return False

    def get_inflation_update(self, **kwargs):
        round_number, self_player, other_player = kwargs["round_number"], kwargs["self_player"], kwargs["other_player"]
        if not self.show_inflation_update:
            return ""

        self_loss = 1 - self_player.delta ** (round_number - 1)
        other_loss = 1 - other_player.delta ** (round_number - 1)
        other_name = other_player.public_name

        if not self_loss and not other_loss:
            return ""

        inflation_update = f"Due to inflation, "
        if self_loss == other_loss:
            inflation_update += ("the money you gain is worth " +
                                 f"{pretty_number(self_loss * 100)}% less than in the first round.")
        else:
            if self_loss:
                inflation_update += f"the money you gain is worth {pretty_number(self_loss * 100)}% " \
                                    f"less than in the first round."
            if other_loss:
                inflation_addition = f"the money {other_name} gains is worth {pretty_number(other_loss * 100)}% " \
                                     f"less than in the first round."
                if self_loss:
                    inflation_addition = inflation_addition.capitalize()
                else:
                    inflation_addition += f" The money you gains is worth the same as in the first round."
                inflation_update += f" {inflation_addition}"
            else:
                inflation_update += f" The money {other_name} gains is worth the same as in the first round."
        return inflation_update + "\n"

    def generate_make_offer_txt(self, **kwargs):
        round_number = kwargs["round_number"]
        other_player_name = kwargs["other_player_name"]
        inflation_update = kwargs["inflation_update"] if "inflation_update" in kwargs else ""
        ask_for_offer = f"Send your offer to divide ${self.money_to_divide:,}" + \
                        (f" and a message to {other_player_name}" if self.messages_allowed else "") + "."
        return f"Round {round_number}\n{inflation_update}".strip() + "\n" + ask_for_offer

    @staticmethod
    def generate_accept_offer_txt(**kwargs):
        round_number = kwargs["round_number"]
        proposer_name = kwargs["proposer_name"]
        proposer_gain = kwargs["proposer_gain"]
        receiver_name = kwargs["receiver_name"]
        receiver_gain = kwargs["receiver_gain"]
        offer_message = kwargs["offer_message"]
        inflation_update = kwargs["inflation_update"] if "inflation_update" in kwargs else ""

        receiver_message_format = f"Answer with {{\"decision\": \"accept\"}} " \
                                  f"or {{\"decision\": \"reject\"}}"
        offer_text = f"\nRound {round_number}\n{inflation_update}" + \
                     f"{proposer_name}'s offer:\n" + \
                     (f"# {proposer_name}'s message: {offer_message} \n"
                      if offer_message is not None else "") + \
                     f"# {receiver_name} gain: {receiver_gain} \n" + \
                     f"# {proposer_name} gain: {proposer_gain} \n" + \
                     f"Do you accept this offer?"
        return receiver_message_format, offer_text

    def make_offer_turn(self, **kwargs):
        proposer, receiver = kwargs["proposer"], kwargs["receiver"]
        round_number = kwargs["round_number"]

        player_inflation_update = self.get_inflation_update(round_number=round_number,
                                                            self_player=proposer,
                                                            other_player=receiver)

        offer_text = self.generate_make_offer_txt(round_number=round_number,
                                                  player_inflation_update=player_inflation_update,
                                                  other_player_name=receiver.public_name)

        proposer.add_to_buffer(offer_text)
        proposer.edit_system_message(proposer.req_offer_text)
        proposer.act(self.is_offer_in_format)
        self.data_logger.add_action(player_name=proposer.public_name,
                                    data=proposer.last_action_json,
                                    round_number=round_number)

    def accept_offer_turn(self, **kwargs):
        proposer, receiver = kwargs["proposer"], kwargs["receiver"]
        round_number = kwargs["round_number"]
        player_inflation_update = self.get_inflation_update(round_number=round_number,
                                                            self_player=receiver,
                                                            other_player=proposer)
        proposer_gain = proposer.last_action_json[f"{self.lower_name(proposer.public_name)}_gain"]
        receiver_gain = proposer.last_action_json[f"{self.lower_name(receiver.public_name)}_gain"]
        offer_message = proposer.last_action_json["message"] if self.messages_allowed else None
        system_message, accept_offer_text = self.generate_accept_offer_txt(round_number=round_number,
                                                                           proposer_name=proposer.public_name,
                                                                           proposer_gain=proposer_gain,
                                                                           receiver_name=receiver.public_name,
                                                                           receiver_gain=receiver_gain,
                                                                           offer_message=offer_message,
                                                                           inflation_update=player_inflation_update)
        receiver.edit_system_message(system_message)
        receiver.add_to_buffer(accept_offer_text)
        receiver.act(self.is_decision_in_format)
        self.data_logger.add_action(player_name=receiver.public_name,
                                    data=receiver.last_action_json,
                                    round_number=round_number)

    @staticmethod
    def generate_response_texts(**kwargs):
        accepted = kwargs["accepted"]
        round_number = kwargs["round_number"]
        proposer_name = kwargs["proposer_name"]
        receiver_name = kwargs["receiver_name"]
        if accepted:
            print(f"{receiver_name} accepted the offer!")
            return f"{receiver_name} accepted the offer!", f"You accepted {proposer_name}'s offer.\n"
        else:
            return f"{receiver_name} rejected your offer from round {round_number}.\n", \
                f"You have chosen to reject {proposer_name}'s offer from round {round_number}.\n"

    def update_round(self, **kwargs):
        proposer, receiver = kwargs["proposer"], kwargs["receiver"]
        round_number = kwargs["round_number"]
        accepted = receiver.last_action_json["decision"] == "accept"
        first_txt, second_txt = self.generate_response_texts(accepted=accepted, round_number=round_number,
                                                             proposer_name=proposer.public_name,
                                                             receiver_name=receiver.public_name)
        proposer.add_to_buffer(first_txt)
        receiver.add_to_buffer(second_txt)
        return accepted

    def game(self):
        for round_number in range(1, self.max_rounds + 1):
            proposer = self.player_1 if round_number % 2 else self.player_2
            receiver = self.player_2 if round_number % 2 else self.player_1
            self.make_offer_turn(proposer=proposer, receiver=receiver, round_number=round_number)
            self.accept_offer_turn(proposer=proposer, receiver=receiver, round_number=round_number)
            if self.update_round(proposer=proposer, receiver=receiver, round_number=round_number):
                break
        else:
            print(f"Game over. There is no agreement after {self.max_rounds} rounds.")
        self.data_logger.save()
        self.player_1.end_chat()
        self.player_2.end_chat()
        self.player_1.clean_history()
        self.player_2.clean_history()
