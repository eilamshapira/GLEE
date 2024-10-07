import os
import json
from consts import OTREE_CONFIGS_PATH

minimum_hit_qualifications = {'QualificationTypeId': '00000000000000000040',  # NumberHitsApproved ID
                              'Comparator': 'GreaterThanOrEqualTo',
                              'IntegerValues': [50],
                              'RequiredToPreview': True}
maximum_hit_qualifications = {'QualificationTypeId': '00000000000000000040',  # NumberHitsApproved ID
                              'Comparator': 'LessThanOrEqualTo',
                              'IntegerValues': [10000],
                              'RequiredToPreview': True}
approval_rate_qualification = {'QualificationTypeId': '000000000000000000L0',  # HIT Approval Rate (%) ID
                               'Comparator': 'GreaterThanOrEqualTo',
                               'IntegerValues': [95],
                               'RequiredToPreview': True}

base_qualifications = [minimum_hit_qualifications, maximum_hit_qualifications, approval_rate_qualification]

BASE_MTURK_HIT_SETTINGS = dict(
    keywords='bonus, study, games, game',
    title='Simple Game',
    description='Simple Game Research Study',
    frame_height=500,
    template='global/mturk_template.html',
    minutes_allotted_per_assignment=20,
    expiration_hours=24,
    qualification_requirements=base_qualifications,
)


def create_full_qualifications(qualification_id):
    new_qualification = [{'QualificationTypeId': qualification_id, 'Comparator': 'DoesNotExist',
                          'RequiredToPreview': True}]
    return base_qualifications + new_qualification


session_configs_temp = []
config_files = os.listdir(OTREE_CONFIGS_PATH)
while config_files:
    filename = config_files.pop()
    filepath = os.path.join(OTREE_CONFIGS_PATH, filename)
    if filename.endswith('.json'):
        with open(filepath, 'r') as file:
            data = json.load(file)
            name_without_extension = os.path.splitext(filename)[0]
            if "sample" in data:  # sample configs
                name_without_config = name_without_extension.split('_')[:-1]
                name = "sample_" + '_'.join(name_without_config)
                display_name = "Sample: " + ' '.join(name_without_config)
                config = dict(
                    name=name,
                    display_name=name,
                    num_demo_participants=1,
                    app_sequence=['otree_game'],
                    path=name_without_extension,
                )
                ordered_config = [0, data['game_type'], 0, 0, config]
                session_configs_temp.append(ordered_config)
                continue

            assert 'participation_fee' in data, "participation_fee is required in the config file, path: " + filepath
            base_mturk_hit_settings = {k: v for k, v in BASE_MTURK_HIT_SETTINGS.items()}
            base_mturk_hit_settings['title'] = data['title']
            base_mturk_hit_settings['description'] = data['description']
            base_mturk_hit_settings['qualification_requirements'] = create_full_qualifications(data['qualification_id'])
            base_mturk_hit_settings['grant_qualification_id'] = data['qualification_id']
            config = dict(
                name=data['name'],
                display_name=data['display_name'],
                num_demo_participants=1,
                app_sequence=['otree_game'],
                path=name_without_extension,
                participation_fee=data['participation_fee'],
                mturk_hit_settings=base_mturk_hit_settings,
            )
            base_qualifications_num = int(data['qualification_base'].split('_')[1])
            config_game_type_id = int(data['name'].split('_')[1])
            ordered_config = [1, data['game_type'], base_qualifications_num, config_game_type_id, config]
            session_configs_temp.append(ordered_config)
    elif os.path.isdir(filepath):
        config_files.extend([os.path.join(filename, f) for f in os.listdir(filepath)])

session_configs_temp.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
SESSION_CONFIGS = [x[4] for x in session_configs_temp]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.0, participation_fee=0.0, doc="",
    mturk_hit_settings=dict(
        keywords='bonus, study, games, game',
        title='Simple Game',
        description='Simple Game Research Study',
        frame_height=500,
        template='global/mturk_template.html',
        minutes_allotted_per_assignment=20,
        expiration_hours=24,
        qualification_requirements=base_qualifications,
    ),
)

PARTICIPANT_FIELDS = ['bot']
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = False
CURRENCY_DECIMAL_PLACES = 2

ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = os.environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """ """

SECRET_KEY = 'otree_secret_key'

# MIDDLEWARE = [
#     'django.middleware.security.SecurityMiddleware',
#     'django.middleware.common.CommonMiddleware',
#     # Add other middleware here
# ]
