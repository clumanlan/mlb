import pandas as pd

from models.n_pa_predictor.processing.pipeline import build_n_pa_label, build_batter_game_frame


def _make_pbp_row(**overrides):
    row = {
        'gamepk': '1',
        'play_id': 'p1',
        'batter_id': '1',
        'play_result': 'Single',
    }
    row.update(overrides)
    return row


def test_build_n_pa_label_counts_one_pa_outcome_per_play_id():
    """Each play_id may appear as multiple pitch rows in pbp — the label
    must count distinct PAs (play_id), not pitch rows."""

    pbp = pd.DataFrame([
        _make_pbp_row(play_id='p1', play_result='Single'),
        _make_pbp_row(play_id='p1', play_result='Single'),  # same PA, 2nd pitch row
        _make_pbp_row(play_id='p2', play_result='Strikeout'),
    ])

    result = build_n_pa_label(pbp)

    row = result[(result['gamepk'] == '1') & (result['batter_id'] == '1')].iloc[0]
    assert row['n_pa'] == 2


def test_build_n_pa_label_counts_a_hit_by_pitch_as_a_pa():
    """HBP is a real plate appearance (ON_BASE_NOT_HIT) — must count."""

    pbp = pd.DataFrame([_make_pbp_row(play_id='p1', play_result='Hit By Pitch')])

    result = build_n_pa_label(pbp)

    assert result.iloc[0]['n_pa'] == 1


def test_build_n_pa_label_counts_a_reach_on_error_as_a_pa():
    """'Field Error' (reached base on a fielding error) is a completed PA —
    the batter put the ball in play — but is missing from hit_predictor's
    shared PBP.PA_OUTCOMES. n_pa_predictor's own PA_OUTCOMES extends it so
    this isn't silently dropped (was undercounting ~1,100 PAs/season)."""

    pbp = pd.DataFrame([_make_pbp_row(play_id='p1', play_result='Field Error')])

    result = build_n_pa_label(pbp)

    assert result.iloc[0]['n_pa'] == 1


def test_build_n_pa_label_counts_a_sac_fly_double_play_as_a_pa():
    """'Sac Fly Double Play' is the same category as 'Sac Fly' (already
    counted) — also missing from hit_predictor's shared PBP.PA_OUTCOMES."""

    pbp = pd.DataFrame([_make_pbp_row(play_id='p1', play_result='Sac Fly Double Play')])

    result = build_n_pa_label(pbp)

    assert result.iloc[0]['n_pa'] == 1


def test_build_n_pa_label_excludes_non_pa_outcome_rows():
    """A play_result outside PBP.PA_OUTCOMES (e.g. a runner-movement event
    with no batting outcome) must not inflate the batter's PA count."""

    pbp = pd.DataFrame([
        _make_pbp_row(play_id='p1', play_result='Single'),
        _make_pbp_row(play_id='p2', play_result='Stolen Base'),
    ])

    result = build_n_pa_label(pbp)

    assert result.iloc[0]['n_pa'] == 1


def test_build_n_pa_label_is_per_batter_game():
    """Two different batters in the same game get separate rows."""

    pbp = pd.DataFrame([
        _make_pbp_row(batter_id='1', play_id='p1', play_result='Single'),
        _make_pbp_row(batter_id='2', play_id='p2', play_result='Groundout'),
    ])

    result = build_n_pa_label(pbp)

    assert set(zip(result['batter_id'], result['n_pa'])) == {('1', 1), ('2', 1)}


def test_build_batter_game_frame_one_row_per_starter_per_game():
    """A starter (has a batting_order) gets exactly one row with their
    realized n_pa attached."""

    pbp = pd.DataFrame([
        _make_pbp_row(batter_id='1', play_id='p1', play_result='Single'),
        _make_pbp_row(batter_id='1', play_id='p2', play_result='Groundout'),
    ])
    batter_boxscore = pd.DataFrame([
        {'gamepk': '1', 'personId': '1', 'batting_order': 3},
    ])
    schedule = pd.DataFrame([
        {'gamepk': '1', 'game_date': pd.Timestamp('2023-05-01')},
    ])

    result = build_batter_game_frame(pbp, batter_boxscore, schedule)

    assert len(result) == 1
    assert result.iloc[0]['n_pa'] == 2
    assert result.iloc[0]['batting_order'] == 3


def test_build_batter_game_frame_excludes_non_starters():
    """A batter with no batting_order (pinch hitter/sub) is excluded —
    scope is starters only, same filter hit_predictor's create_pa_outcome
    applies via its inner join on batting_order."""

    pbp = pd.DataFrame([
        _make_pbp_row(batter_id='1', play_id='p1', play_result='Single'),
        _make_pbp_row(batter_id='2', play_id='p2', play_result='Single'),
    ])
    batter_boxscore = pd.DataFrame([
        {'gamepk': '1', 'personId': '1', 'batting_order': 3},
        # batter '2' has no batting_order row -> pinch hitter/sub
    ])
    schedule = pd.DataFrame([
        {'gamepk': '1', 'game_date': pd.Timestamp('2023-05-01')},
    ])

    result = build_batter_game_frame(pbp, batter_boxscore, schedule)

    assert list(result['batter_id']) == ['1']


def test_build_batter_game_frame_includes_game_date():
    """game_date (from schedule) must survive so downstream rolling
    features can sort/join on it."""

    pbp = pd.DataFrame([_make_pbp_row(batter_id='1', play_id='p1', play_result='Single')])
    batter_boxscore = pd.DataFrame([{'gamepk': '1', 'personId': '1', 'batting_order': 3}])
    schedule = pd.DataFrame([{'gamepk': '1', 'game_date': pd.Timestamp('2023-05-01')}])

    result = build_batter_game_frame(pbp, batter_boxscore, schedule)

    assert result.iloc[0]['game_date'] == pd.Timestamp('2023-05-01')
