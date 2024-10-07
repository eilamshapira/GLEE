-- We split the otree_game_player data due to the large amount of rows
SELECT *
from otree_game_player
where id_in_group < 15

SELECT *
from otree_game_player
where id_in_group >= 15

SELECT *
from otree_participant INNER JOIN otree_session on otree_session.id = otree_participant.session_id