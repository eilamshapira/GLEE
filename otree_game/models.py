from otree.api import models, BaseConstants, BaseSubsession, BaseGroup, BasePlayer
from consts import OTREE_MAX_ROUNDS, OTREE_CONFIGS_PATH, OTREE_PAGES


# ------------- OTREE MODELS -------------

class Constants(BaseConstants):
    name_in_url = 'GLEE'
    players_per_group = None
    num_rounds = OTREE_MAX_ROUNDS
    base_config_path = OTREE_CONFIGS_PATH
    pages_path = OTREE_PAGES

    quiz_word = 'sdkot'
    instructions_quiz = (f'In the comment text box below, please type "{quiz_word}" (without commas and quotes), '
                         'so we can be sure you are reading this. '
                         'If you fail to do so, you will be unable to complete this HIT.')
    quiz_failed = 'FAILED'
    quiz_failed_start = quiz_failed + '_START'
    quiz_failed_end = quiz_failed + '_END'
    quiz_success = 'SUCCESS'
    quiz_empty = 'EMPTY'


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):  # it's important to distinguish between the Player otree model and the otree_player class
    config_path = models.StringField(doc="Path to the configuration file")
    player_name = models.StringField(doc="Player name")
    real_turn = models.IntegerField(doc="Real turn number")
    who_propose = models.StringField(doc="Who is the proposer - human or bot")
    offer = models.IntegerField(min=0, blank=True, doc="The offer made by the proposer")
    proposer_message = models.StringField(initial="", blank=True, doc="Message from the proposer")
    proposer_recommendation = models.BooleanField(blank=True, doc="Recommendation for the offer made by the proposer")
    receiver_message = models.StringField(initial="", blank=True, doc="Message from the receiver")
    accepted = models.IntegerField(initial=0, doc="Receiver accepted the offer")
    utility = models.FloatField(initial=0, doc="Player utility")
    additional_info = models.StringField(initial="", blank=True, doc="Additional information")
    time_spent_on_action_page = models.FloatField(initial=0, doc="Time spent on the action page")
    show_instructions = models.BooleanField(initial=True, doc="Show instructions on action page")
    quiz_answer = models.StringField(blank=True, initial="", doc="Quiz answer")

