
class PBP:
        
    # dropped entirely - duplicates or not needed
    DROP_COLS = {
        'batter_player_name',
        'pitcher_player_name',
    }

    # never features - just keys for joining/grouping
    INDEX_COLS = {
        'gamepk',
        'play_id',
        'event_index',
        'pitch_number',
    }

    # entity keys - used for aggregations and lookups
    ENTITY_COLS = {
        'batter_id',
        'batter_name',
        'batter_team_id',
        'batter_team_name',
        'pitcher_id',
        'pitcher_name',
        'pitcher_team_id',
        'pitcher_team_name',
    }

    # game state context - condition features or slice filters
    CONTEXT_COLS = {
        'inning',
        'half_inning',
        'count_balls',
        'count_strikes',
        'count_outs',
    }

    # target / label adjacent - what happened
    OUTCOME_COLS = {
        'play_result',
        'is_pitch',
        'is_in_play',
        'is_strike',
        'is_ball',
        'is_out',
        'pitch_call',
    }

    # raw pitch features - inputs to stuff/movement models
    PITCH_PHYSICS_COLS = {
        'pitch_type',
        'start_speed',
        'end_speed',
        'extension',
        'plate_time',
        'spin_rate',
        'spin_direction',
        'pfx_x',
        'pfx_z',
        'break_vertical',
        'break_vertical_induced',
        'break_horizontal',
        'break_angle',
        'break_length',
        'break_y',
        'release_pos_x',
        'release_pos_y',
        'release_pos_z',
        'vx0',
        'vy0',
        'vz0',
        'ax',
        'ay',
        'az',
    }

    # location features - inputs to location/command models
    PITCH_LOCATION_COLS = {
        'plate_x',
        'plate_z',
        'zone',
        'strike_zone_top',
        'strike_zone_bottom',
        'sz_x',
        'sz_y',
        'type_confidence',
    }

    # batted ball features - null for non-contact, inputs to xBA/xSLG
    BATTED_BALL_COLS = {
        'launch_speed',
        'launch_angle',
        'total_distance',
        'trajectory',
        'hardness',
        'hit_location',
        'hit_coord_x',
        'hit_coord_y',
    }

    # play result bins 
    HITS = {"Single", "Double", "Triple", "Home Run"}

    OUTS = {
        "Strikeout", "Strikeout Double Play",
        "Groundout", "Forceout", "Grounded Into DP", "Double Play", 
        "Triple Play", "Fielders Choice Out", "Field Out",
        "Flyout", "Pop Out", "Sac Fly",
        "Lineout", "Bunt Groundout", "Bunt Pop Out", "Bunt Lineout",
        "Fielders Choice", 'Sac Bunt'
    }

    ON_BASE_NOT_HIT = {"Walk", "Intent Walk", "Hit By Pitch", "Catcher Interference"}

    ERROR = 'Field Error'

    # For pitcher PA denominator
    PA_OUTCOMES = HITS | OUTS | ON_BASE_NOT_HIT
