from .models import Constants, Subsession, Group, Player
from .pages import (EnterDetails, Introduction, ProposerPage, LLMActionPage, ReceiverPage, ResponsePage,
                    FinalQuiz, Results, QuizFailed)
# FIXME synchronize the data (mainly in the connection between analysis and otree)
# FIXME copy process data to...
doc = """
GLEE - Games in Language-based Economic Environments
"""

page_sequence = [EnterDetails, Introduction,  # Start pages
                 ProposerPage, LLMActionPage, ReceiverPage, ResponsePage,  # Game pages
                 FinalQuiz, Results, QuizFailed]  # End pages
