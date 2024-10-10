from .models import Constants, Subsession, Group, Player
from .pages import (EnterDetails, Introduction, ProposerPage, LLMActionPage, ReceiverPage, ResponsePage,
                    FinalQuiz, Results, QuizFailed)
# ADD documentation
# FIXME synchronize the data read and use,
#  mainly in the connection between analysis and the LLM and otree environments (including process data)
doc = """
GLEE - Games in Language-based Economic Environments
"""

page_sequence = [EnterDetails, Introduction,  # Start pages
                 ProposerPage, LLMActionPage, ReceiverPage, ResponsePage,  # Game pages
                 FinalQuiz, Results, QuizFailed]  # End pages
