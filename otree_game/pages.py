from otree.api import Page
import time
import random
from utils.utils import pretty_number

from .models import Player, Constants
from .otree_utils import (load_bot_name, create_bot, load_bot, save_round,
                          pretty_rules_text, pretty_html_text, complete_code_hash)
# ADD replace Alice and Bob with not constant names


# ------------- PAGES -------------

class EnterDetails(Page):
    template_name = f'{Constants.pages_path}//EnterDetails.html'
    form_model = 'player'
    form_fields = ['player_name']

    @staticmethod
    def is_displayed(player: Player):
        player.quiz_answer = Constants.quiz_empty
        player.time_spent_on_action_page = 0
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        bot_name = load_bot_name(player.session.config['path']).lower()
        return {
            'bot_name': bot_name,
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        player.participant.vars['bot'] = create_bot(player.session.config['path'], human_name=player.player_name)


class Introduction(Page):
    template_name = f'{Constants.pages_path}//Introduction.html'
    form_model = 'player'
    form_fields = ['quiz_answer']

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        rules = bot.player_1_rules if bot.human_start else bot.player_2_rules
        return {
            'rules': pretty_rules_text(rules),
            'bonus_text': pretty_html_text(bot.get_bonus_text()),
            'is_bonus_text': bot.get_bonus_text() != '',
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        if player.quiz_answer is None:
            player.quiz_answer = Constants.quiz_failed_start
        else:
            player.quiz_answer = player.quiz_answer.strip()
            if player.quiz_answer != Constants.quiz_word:
                player.quiz_answer = Constants.quiz_failed_start


class ProposerPage(Page):
    template_name = f'{Constants.pages_path}//ProposerPage.html'
    form_model = 'player'
    form_fields = ['offer', 'proposer_message', 'proposer_recommendation', 'show_instructions']

    @staticmethod
    def is_displayed(player: Player):
        if player.quiz_answer.startswith(Constants.quiz_failed):
            return False
        bot = load_bot(player)
        res = bot.show_proposer_page(player)
        return res

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        fields = bot.get_otree_player_fields()
        rules = bot.player_1_rules if bot.human_start else bot.player_2_rules
        if player.time_spent_on_action_page == 0:
            player.time_spent_on_action_page = - time.time()
        return {
            'is_offer': 'offer' in fields,
            'is_proposer_message': 'proposer_message' in fields,
            'is_proposer_recommendation': 'proposer_recommendation' in fields,
            'proposer_text': pretty_html_text(bot.get_proposer_text(player)),
            'offer_type_slider': bot.get_offer_type() == 'slider',
            'max_offer': bot.get_max_offer() if 'offer' in fields else None,
            'bot_name': bot.bot_player.public_name,
            'instructions': pretty_html_text(rules),
            'show_instructions': int(bot.show_instructions),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        if player.time_spent_on_action_page < 0:
            player.time_spent_on_action_page += time.time()
        bot = load_bot(player)
        bot.show_instructions = player.show_instructions


class LLMActionPage(Page):
    template_name = f'{Constants.pages_path}//LLMActionPage.html'

    @staticmethod
    def is_displayed(player: Player):
        if player.quiz_answer.startswith(Constants.quiz_failed):
            return False
        bot = load_bot(player)
        return bot.show_proposer_page(player) or bot.show_receiver_page(player)

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        return {
            'bot_name': bot.bot_player.public_name,
        }

    @staticmethod
    def before_next_page(player, timeout_happened):
        bot = load_bot(player)
        if bot.show_proposer_page(player):
            bot.bot_receiver_turn(player)
        if bot.show_receiver_page(player):
            bot.bot_proposer_turn(player)


class ReceiverPage(Page):
    template_name = f'{Constants.pages_path}//ReceiverPage.html'
    form_model = 'player'
    form_fields = ['receiver_message', 'accepted', 'show_instructions']

    @staticmethod
    def is_displayed(player: Player):
        if player.quiz_answer.startswith(Constants.quiz_failed):
            return False
        bot = load_bot(player)
        res = bot.show_receiver_page(player)
        return res

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        special_decision = bot.special_decision()
        rules = bot.player_1_rules if bot.human_start else bot.player_2_rules
        if player.time_spent_on_action_page == 0:
            player.time_spent_on_action_page = - time.time()
        return {
            'is_receiver_message': 'receiver_message' in bot.get_otree_player_fields(),
            'receiver_text': pretty_html_text(bot.get_receiver_text(player)),
            'is_special_decision': special_decision is not None,
            'value_button_text_color_list': special_decision,
            'instructions': pretty_html_text(rules),
            'show_instructions': int(bot.show_instructions),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        if player.time_spent_on_action_page < 0:
            player.time_spent_on_action_page += time.time()
        bot = load_bot(player)
        bot.show_instructions = player.show_instructions


class ResponsePage(Page):
    template_name = f'{Constants.pages_path}//ResponsePage.html'

    @staticmethod
    def is_displayed(player: Player):
        if player.quiz_answer.startswith(Constants.quiz_failed):
            return False
        bot = load_bot(player)
        return bot.show_response_page()

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        return {
            'response_text': pretty_html_text(bot.get_response_text(player)),
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        save_round(player)


class FinalQuiz(Page):
    template_name = f'{Constants.pages_path}//FinalQuiz.html'
    form_model = 'player'
    form_fields = ['quiz_answer']

    @staticmethod
    def is_displayed(player: Player):
        if player.quiz_answer.startswith(Constants.quiz_failed):
            return False
        bot = load_bot(player)
        return bot.show_results_and_set_payoffs(player)

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        question_text, right_answer, wrong_answers = bot.final_quiz_options()
        right_answer = pretty_number(right_answer)
        wrong_answers = [pretty_number(wrong_answer) for wrong_answer in wrong_answers]
        wrong_answers = list(set(wrong_answers))

        if right_answer in wrong_answers:
            wrong_answers.remove(right_answer)
        options = [right_answer] + wrong_answers
        random.shuffle(options)
        options = [(i, option) for i, option in enumerate(options)]
        return {
            'options': options,
            'question_text': question_text,
        }

    @staticmethod
    def before_next_page(player: Player, timeout_happened: bool):
        bot = load_bot(player)
        _, right_answer, _ = bot.final_quiz_options()
        right_answer = pretty_number(right_answer)
        if player.quiz_answer is None:
            player.quiz_answer = Constants.quiz_failed_end
        elif str(player.quiz_answer) != str(right_answer):
            player.quiz_answer = Constants.quiz_failed_end
        else:
            player.quiz_answer = Constants.quiz_success


class Results(Page):
    template_name = f'{Constants.pages_path}//Results.html'

    @staticmethod
    def is_displayed(player: Player):
        return player.quiz_answer == Constants.quiz_success

    @staticmethod
    def vars_for_template(player: Player):
        bot = load_bot(player)
        final_text = pretty_html_text(bot.get_final_text(player))
        return {
            'final_text': final_text,
            'is_final_text': final_text != '',
            'complete_code': complete_code_hash(player),
        }


class QuizFailed(Page):
    template_name = f'{Constants.pages_path}//QuizFailed.html'

    @staticmethod
    def is_displayed(player: Player):
        return player.quiz_answer.startswith(Constants.quiz_failed)

    @staticmethod
    def vars_for_template(player: Player):
        return {
            'is_start': player.quiz_answer == Constants.quiz_failed_start,
        }
