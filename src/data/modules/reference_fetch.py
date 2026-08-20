import os
import time
import tempfile
import logging
import pandas as pd
import statsapi
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

PLAYER_INFO_S3_KEY = 'reference/player_info/player_info.parquet'
TEAM_INFO_S3_KEY = 'reference/team_info/team_info.parquet'


class ReferenceFetcher:
    def __init__(self, aws_bucket: str = 'mlbdk', aws_region: str = 'us-east-2'):
        self.aws_bucket = aws_bucket
        self.aws_region = aws_region
        self.s3_client = boto3.client('s3', region_name=aws_region)

    # --- S3 helpers ---

    def _upload_to_s3(self, df: pd.DataFrame, s3_key: str):
        try:
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                df.to_parquet(tmp.name, index=False)
                self.s3_client.upload_file(tmp.name, self.aws_bucket, s3_key)
            os.unlink(tmp.name)
            logger.info(f"Uploaded {len(df)} rows to s3://{self.aws_bucket}/{s3_key}")
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

    def _read_from_s3(self, s3_key: str) -> pd.DataFrame:
        import io
        obj = self.s3_client.get_object(Bucket=self.aws_bucket, Key=s3_key)
        return pd.read_parquet(io.BytesIO(obj['Body'].read()))

    # --- Player info ---

    def fetch_players_for_season(self, season: int) -> pd.DataFrame:
        """
        Fetch all players for a given season using the sports_players bulk endpoint.
        Returns one row per player with standardized columns.
        """
        logger.info(f"Fetching players for season {season}")
        try:
            response = statsapi.get('sports_players', {'season': season, 'sportId': 1, 'gameType': 'R'})
            players = response.get('people', [])
        except Exception as e:
            logger.error(f"Failed to fetch players for season {season}: {e}")
            return pd.DataFrame()

        rows = []
        for p in players:
            rows.append({
                'person_id':          p['id'],
                'player_name':        p.get('fullName'),
                'birthDate':          p.get('birthDate'),
                'birthCountry':       p.get('birthCountry'),
                'height':             p.get('height'),
                'weight':             p.get('weight'),
                'primaryPosition':    p.get('primaryPosition', {}).get('name'),
                'primaryPositionType':p.get('primaryPosition', {}).get('type'),
                'draftYear':          p.get('draftYear'),
                'mlbDebutDate':       p.get('mlbDebutDate'),
                'strikeZoneTop':      p.get('strikeZoneTop'),
                'strikeZoneBottom':   p.get('strikeZoneBottom'),
                'batSide':            p.get('batSide', {}).get('code'),
                'pitchHand':          p.get('pitchHand', {}).get('code'),
            })

        logger.info(f"Season {season}: {len(rows)} players fetched")
        return pd.DataFrame(rows)

    def _fetch_players_by_ids(self, person_ids: list) -> pd.DataFrame:
        """
        Direct lookup for player IDs not found in the season roster bulk endpoint.
        Uses the people endpoint with a comma-separated list of IDs.
        """
        ids_str = ','.join(str(i) for i in person_ids)
        try:
            response = statsapi.get('people', {'personIds': ids_str})
            players = response.get('people', [])
        except Exception as e:
            logger.error(f"Failed direct player lookup for {ids_str}: {e}")
            return pd.DataFrame()

        rows = []
        for p in players:
            rows.append({
                'person_id':           p['id'],
                'player_name':         p.get('fullName'),
                'birthDate':           p.get('birthDate'),
                'birthCountry':        p.get('birthCountry'),
                'height':              p.get('height'),
                'weight':              p.get('weight'),
                'primaryPosition':     p.get('primaryPosition', {}).get('name'),
                'primaryPositionType': p.get('primaryPosition', {}).get('type'),
                'draftYear':           p.get('draftYear'),
                'mlbDebutDate':        p.get('mlbDebutDate'),
                'strikeZoneTop':       p.get('strikeZoneTop'),
                'strikeZoneBottom':    p.get('strikeZoneBottom'),
                'batSide':             p.get('batSide', {}).get('code'),
                'pitchHand':           p.get('pitchHand', {}).get('code'),
            })
        return pd.DataFrame(rows)

    def backfill_historical_players(self, start: int = 2000, end: int = 2025,
                                    save_to_s3: bool = True) -> pd.DataFrame:
        """
        Fetch all players across seasons start–end, dedupe by person_id, and save
        to S3 at reference/player_info/player_info.parquet.
        """
        all_seasons = []
        for season in range(start, end + 1):
            df = self.fetch_players_for_season(season)
            if not df.empty:
                all_seasons.append(df)
            time.sleep(1.1)

        if not all_seasons:
            logger.warning("No player data fetched across any season.")
            return pd.DataFrame()

        combined = pd.concat(all_seasons, ignore_index=True)
        combined = combined.drop_duplicates(subset='person_id').reset_index(drop=True)
        logger.info(f"Backfill complete: {len(combined)} unique players ({start}-{end})")

        if save_to_s3:
            self._upload_to_s3(combined, PLAYER_INFO_S3_KEY)

        return combined

    def upsert_players(self, person_ids: list, season: int) -> pd.DataFrame:
        """
        Check person_ids against existing player_info in S3. Fetch and upsert any new ones.

        Called from daily_mlb_fetch after all tables are fetched, passing all person_ids
        from today's batter boxscore, pitcher boxscore, and play-by-play.

        Returns the updated player_info DataFrame.
        """
        # Load existing player_info
        try:
            existing = self._read_from_s3(PLAYER_INFO_S3_KEY)
            existing_ids = set(existing['person_id'].astype(str))
            logger.info(f"Existing player_info: {len(existing)} players")
        except self.s3_client.exceptions.NoSuchKey:
            logger.warning("No existing player_info found — starting fresh")
            existing = pd.DataFrame()
            existing_ids = set()

        new_ids = set(str(p) for p in person_ids) - existing_ids

        if not new_ids:
            logger.info("No new players to upsert")
            return existing

        logger.info(f"{len(new_ids)} new players found — fetching season {season} roster")
        season_df = self.fetch_players_for_season(season)

        new_players = season_df[season_df['person_id'].astype(str).isin(new_ids)]
        logger.info(f"{len(new_players)} new players matched from season roster")

        # Fallback: look up any IDs not in the season roster directly via the people endpoint
        still_missing = new_ids - set(new_players['person_id'].astype(str))
        if still_missing:
            logger.info(f"{len(still_missing)} IDs not in season roster — looking up directly: {still_missing}")
            direct = self._fetch_players_by_ids(list(still_missing))
            if not direct.empty:
                new_players = pd.concat([new_players, direct], ignore_index=True)
                logger.info(f"{len(direct)} additional players fetched via direct lookup")

        if new_players.empty:
            logger.warning(f"New IDs not found anywhere: {new_ids}")
            return existing

        updated = pd.concat([existing, new_players], ignore_index=True)
        updated = updated.drop_duplicates(subset='person_id').reset_index(drop=True)

        self._upload_to_s3(updated, PLAYER_INFO_S3_KEY)
        return updated

    # --- Team info ---

    def fetch_team_info(self, start: int = 2000, end: int = 2025,
                        save_to_s3: bool = True) -> pd.DataFrame:
        """
        Fetch team info for each season start–end, dedupe by (id, season), and save
        to S3 at reference/team_info/team_info.parquet.
        """
        team_list = []
        season_errors = []

        for season in range(start, end + 1):
            try:
                response = statsapi.get('teams', {'season': season, 'sportId': 1})
                df = pd.json_normalize(response['teams'])
                team_list.append(df)
                time.sleep(1.1)
            except Exception as e:
                season_errors.append(season)
                logger.error(f"Error fetching teams for season {season}: {e}")

        if not team_list:
            raise ValueError("No team data successfully fetched.")

        rel_cols = [
            'id', 'name', 'season', 'teamCode', 'abbreviation', 'teamName',
            'shortName', 'league.id', 'league.name', 'division.id', 'division.name',
        ]
        teams_df = pd.concat(team_list, ignore_index=True)
        teams_df = teams_df[[c for c in rel_cols if c in teams_df.columns]]
        teams_df = teams_df.drop_duplicates(['id', 'season']).reset_index(drop=True)

        if season_errors:
            logger.warning(f"{len(season_errors)} seasons failed: {season_errors}")

        logger.info(f"Team backfill complete: {len(teams_df)} rows")

        if save_to_s3:
            self._upload_to_s3(teams_df, TEAM_INFO_S3_KEY)

        return teams_df
