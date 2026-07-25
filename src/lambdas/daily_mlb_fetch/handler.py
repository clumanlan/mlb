from fetchers import fetch_schedule_df, fetch_game_info, fetch_batter_pitcher_boxscore, fetch_playbyplay_data, send_metrics_to_cloudwatch
from reference_fetch import ReferenceFetcher
from status_writer import write_status
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    date = event.get('date') or (datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d')
    year = datetime.now().year
    s3_prefix = 'test' if event.get('env') == 'test' else ''

    results = {}
    started_at = datetime.utcnow()
    games_processed = {}

    try:
        return _run(event, context, date, year, s3_prefix, results, started_at, games_processed)
    except Exception as e:
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_mlb_fetch",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=str(e),
        )
        raise


def _run(event, context, date, year, s3_prefix, results, started_at, games_processed):
    # Step 1: Fetch schedule
    schedule_df, schedule_errors = fetch_schedule_df(date, s3_prefix=s3_prefix)
    results['schedule'] = {
        'games_found': len(schedule_df),
        'errors': len(schedule_errors)
    }
    
    if schedule_df.empty:
        results['message'] = 'No games on schedule date,'
        return {'statusCode': 200, 'body': json.dumps(results)}
    
    logger.info(f"Processing date: {date}")
    logger.info(f"Schedule fetched: {len(schedule_df)} games, {len(schedule_errors)} errors")

    # Extract gamepks from schedule
    gamepks = schedule_df['game_id'].tolist()
    
    # Fetch game info
    game_df, game_errors = fetch_game_info(gamepks, game_date=date, s3_prefix=s3_prefix)
    results['game_info'] = {
        'games_processed': len(game_df['gamePk'].unique()) if not game_df.empty else 0,
        'errors': len(game_errors)
    }

    logger.info(f"Game info fetched: {len(game_df['gamePk'].unique())} games, {len(game_errors)} errors")

    
    # Extract boxscore
    try:
        batter_df, pitcher_df, boxscore_errors = fetch_batter_pitcher_boxscore(gamepks, game_date=date, s3_prefix=s3_prefix)
        results['boxscore'] = {
            'batters_boxscore_processed': len(batter_df['gamePk'].unique()),
            'pitchers_boxscore_processed': len(pitcher_df['gamePk'].unique()),
            'errors': len(boxscore_errors)
        }
        logger.info(f"Boxscores fetched: {len(batter_df['gamePk'].unique())} batter games, {len(pitcher_df['gamePk'].unique())} pitcher games, {len(boxscore_errors)} errors")

    except KeyError as e:
        logger.error(f"KeyError in fetch_batter_pitcher_boxscore: {e}")
        results['boxscore'] = {
            'batters_boxscore_processed': 0,
            'pitchers_boxscore_processed': 0,
            'errors': 1,
            'error_message': f"Missing column: {str(e)}"
        }

    playbyplay_df, playbyplay_errors = fetch_playbyplay_data(gamepks, game_date=date, s3_prefix=s3_prefix)

    results['playbyplay'] = {
        'games_processed': len(playbyplay_df['gamepk'].unique()),
        'errors': len(playbyplay_errors)
    }

    logger.info(f"Play-by-play fetched: {len(playbyplay_df['gamepk'].unique())} games, {len(playbyplay_errors)} errors")

    # Upsert any new players into reference/player_info
    try:
        person_ids = []
        if not batter_df.empty:
            person_ids.extend(batter_df['personId'].dropna().astype(int).tolist())
        if not pitcher_df.empty:
            person_ids.extend(pitcher_df['personId'].dropna().astype(int).tolist())
        if not playbyplay_df.empty:
            person_ids.extend(playbyplay_df['batter_id'].dropna().astype(int).tolist())
            person_ids.extend(playbyplay_df['pitcher_id'].dropna().astype(int).tolist())

        person_ids = list(set(person_ids))
        ref = ReferenceFetcher()
        updated_players = ref.upsert_players(person_ids, year)
        results['player_info'] = {'total_players': len(updated_players)}
        logger.info(f"Player info upsert complete: {len(updated_players)} total players")
    except Exception as e:
        logger.error(f"Failed to upsert player info: {e}")
        results['player_info'] = {'error': str(e)}

    # Log final results
    logger.info(f"Final results: {json.dumps(results, indent=2)}")

    # Send metrics to CloudWatch
    send_metrics_to_cloudwatch(results, date)

    games_processed.update({
        'schedule':        len(schedule_df),
        'game_info':       len(game_df['gamePk'].unique()) if not game_df.empty else 0,
        'batter_boxscore': len(batter_df['gamePk'].unique()) if not batter_df.empty else 0,
        'pitcher_boxscore': len(pitcher_df['gamePk'].unique()) if not pitcher_df.empty else 0,
        'playbyplay':      len(playbyplay_df['gamepk'].unique()) if not playbyplay_df.empty else 0,
    })

    completed_at = datetime.utcnow()
    write_status(
        function_name="daily_mlb_fetch",
        run_date=date,
        status="success",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_seconds=(completed_at - started_at).total_seconds(),
        games_processed=games_processed,
        error=None,
    )

    return {
        'statusCode': 200,
        'body': json.dumps(results)
    }

