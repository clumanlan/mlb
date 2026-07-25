import io
import json
import logging
import tempfile
import os
from datetime import datetime, timedelta

import boto3
import pandas as pd

from preprocessing import (
    process_schedule, process_game_info,
    process_batter_boxscore, process_pitcher_boxscore, process_player_info,
    prepare_batter_boxscore, prepare_pitcher_boxscore, prepare_playbyplay,
)
from status_writer import write_status

logger = logging.getLogger()
logger.setLevel(logging.INFO)

BUCKET = 'mlbdk'
REGION = 'us-east-2'


def _read_s3(s3, key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        raise RuntimeError(f"Failed to read s3://{BUCKET}/{key}: {type(e).__name__}: {e}") from e


def _write_s3(s3, df, key):
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
        s3.upload_file(tmp.name, BUCKET, key)
    os.unlink(tmp.name)
    logger.info(f"Wrote {len(df)} rows to s3://{BUCKET}/{key}")


def lambda_handler(event, context):
    date = event.get('date') or (datetime.now() + timedelta(days=-1)).strftime('%Y-%m-%d')
    year = date[:4]
    s3_prefix = 'test' if event.get('env') == 'test' else ''
    prefix = f"{s3_prefix}/" if s3_prefix else ''

    s3 = boto3.client('s3', region_name=REGION)
    results = {}
    started_at = datetime.utcnow()
    games_processed = {}

    try:
        # --- Read raw data ---
        logger.info(f"Reading raw data for {date}")
        try:
            schedule_raw    = _read_s3(s3, f"{prefix}raw_data/games/schedule/{year}/{date}.parquet")
            game_info_raw   = _read_s3(s3, f"{prefix}raw_data/games/game_info/{year}/{date}.parquet")
            batter_raw      = _read_s3(s3, f"{prefix}raw_data/games/batter_boxscore/{year}/{date}.parquet")
            pitcher_raw     = _read_s3(s3, f"{prefix}raw_data/games/pitcher_boxscore/{year}/{date}.parquet")
            playbyplay_raw  = _read_s3(s3, f"{prefix}raw_data/playbyplay/{year}/{date}.parquet")
            player_info_raw = _read_s3(s3, "reference/player_info/player_info.parquet")
        except RuntimeError as e:
            logger.error(f"S3 read failed: {e}")
            raise

        # --- Process (clean each table independently) ---
        logger.info("Processing raw tables")
        schedule    = process_schedule(schedule_raw)
        game_info   = process_game_info(game_info_raw)
        batter      = process_batter_boxscore(batter_raw)
        pitcher     = process_pitcher_boxscore(pitcher_raw)
        player_info = process_player_info(player_info_raw)

        results['process'] = {
            'schedule_rows':    len(schedule),
            'game_info_rows':   len(game_info),
            'batter_rows':      len(batter),
            'pitcher_rows':     len(pitcher),
            'player_info_rows': len(player_info),
        }
        logger.info(f"Process step complete: {results['process']}")

        # --- Prepare (join tables together) ---
        logger.info("Preparing joined tables")
        batter_prepared     = prepare_batter_boxscore(batter, player_info, schedule)
        pitcher_prepared    = prepare_pitcher_boxscore(pitcher, player_info, schedule)
        playbyplay_prepared = prepare_playbyplay(playbyplay_raw, player_info, schedule)

        results['prepare'] = {
            'batter_rows':     len(batter_prepared),
            'pitcher_rows':    len(pitcher_prepared),
            'playbyplay_rows': len(playbyplay_prepared),
        }
        logger.info(f"Prepare step complete: {results['prepare']}")

        # --- gamePk completeness check ---
        schedule_gamepks = set(schedule['gamepk'].astype(str))
        gamepk_gaps = {}
        for table_name, df in [
            ('batter_prepared',     batter_prepared),
            ('pitcher_prepared',    pitcher_prepared),
            ('playbyplay_prepared', playbyplay_prepared),
        ]:
            missing = schedule_gamepks - set(df['gamepk'].astype(str))
            if missing:
                logger.warning(f"gamePk completeness gap in {table_name}: missing {sorted(missing)}")
                gamepk_gaps[table_name] = sorted(missing)
        if gamepk_gaps:
            results['gamepk_gaps'] = gamepk_gaps
            logger.warning(f"gamePk gaps detected: {gamepk_gaps}")

        # --- Write processed data to S3 ---
        logger.info("Writing stage 1 (processed) tables to S3")
        _write_s3(s3, schedule,  f"{prefix}processed_data/games/schedule/{year}/{date}.parquet")
        _write_s3(s3, game_info, f"{prefix}processed_data/games/game_info/{year}/{date}.parquet")
        _write_s3(s3, batter,    f"{prefix}processed_data/games/batter_boxscore/{year}/{date}.parquet")
        _write_s3(s3, pitcher,   f"{prefix}processed_data/games/pitcher_boxscore/{year}/{date}.parquet")

        logger.info("Writing stage 2 (prepared) tables to S3")
        _write_s3(s3, batter_prepared,     f"{prefix}processed_data/prepared/batter_boxscore/{year}/{date}.parquet")
        _write_s3(s3, pitcher_prepared,    f"{prefix}processed_data/prepared/pitcher_boxscore/{year}/{date}.parquet")
        _write_s3(s3, playbyplay_prepared, f"{prefix}processed_data/prepared/playbyplay/{year}/{date}.parquet")

        games_processed = {
            'schedule':          len(schedule['gamepk'].unique()),
            'batter_prepared':   len(batter_prepared['gamepk'].unique()),
            'pitcher_prepared':  len(pitcher_prepared['gamepk'].unique()),
            'playbyplay_prepared': len(playbyplay_prepared['gamepk'].unique()),
        }

        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_process_data",
            run_date=date,
            status="success",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=None,
        )

        logger.info(f"Final results: {json.dumps(results, indent=2)}")
        return {
            'statusCode': 200,
            'body': json.dumps(results)
        }

    except Exception as e:
        completed_at = datetime.utcnow()
        write_status(
            function_name="daily_process_data",
            run_date=date,
            status="failed",
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=(completed_at - started_at).total_seconds(),
            games_processed=games_processed,
            error=str(e),
        )
        raise