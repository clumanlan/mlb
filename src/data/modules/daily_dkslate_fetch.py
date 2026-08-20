from draft_kings.data import Sport
from draft_kings import Client
from datetime import datetime
import tqdm
import pandas as pd
import re 
import boto3
import tempfile
import os

client = Client()


def fetch_dk_contests(date):
    rel_contests = []
    contests = client.contests(sport=Sport.MLB)

    for contest in contests.contests:
        if contest.starts_at.strftime('%Y-%m-%d')==date:
            rel_contests.append(contest)

    return rel_contests

def fetch_dk_draft_groups(contests):
    
    draft_groups = []
    for contest in tqdm.tqdm(contests):
        draft_group_id = contest.draft_group_id
        if draft_group_id not in draft_groups:
            draft_groups.append(draft_group_id)

    return draft_groups

def map_dk_draft_group_to_game_type(draft_group_ids):

    draft_group_game_type = {}
    for draft_group_id in tqdm.tqdm(draft_group_ids):
        game_type_id = client.draft_group_details(draft_group_id=draft_group_id).contest_details.type_id

        draft_group_game_type[draft_group_id] = game_type_id

    return draft_group_game_type

def filter_dk_contests(contests, draft_groups):

    def contest_to_dict(contest):
        """Convert contest object to dictionary"""
        return {
            'contest_id': contest.contest_id,
            'draft_group_id': contest.draft_group_id,
            'name': contest.name,
            'entry_fee': contest.entries_details.fee,
            'max_entries': contest.entries_details.maximum,
            'current_entries': contest.entries_details.total,
            'payout': contest.payout,
            'starts_at': contest.starts_at,
            'is_double_up': contest.is_double_up,
            'is_fifty_fifty': contest.is_fifty_fifty,
            'is_head_to_head': contest.is_head_to_head,
            'is_guaranteed': contest.is_guaranteed,
            'fantasy_player_points': contest.fantasy_player_points
        }
    
    def extract_contest_type(contest_name):
        """Extract contest type from parentheses in contest name"""
        if pd.isna(contest_name):
            return None
        
        # Look for text in parentheses at the end
        match = re.search(r'\(([^)]+)\)$', contest_name.strip())
        if match:
            return match.group(1).lower()  # Return "night", "early", "turbo", etc.
        else:
            return "classic"  # No parentheses = classic/regular

     
    classic_contests = []
    for contest in contests:
        if contest.draft_group_id in draft_groups:
            classic_contests.append(contest_to_dict(contest))
    
    classic_contests = pd.DataFrame(classic_contests)
    classic_contests['contest_type'] = classic_contests['name'].apply(extract_contest_type)

    return classic_contests


def fetch_dk_competitions_and_players(draft_groups):

    def process_draft_group_competitions(competitions):

        data = []

        for competition in competitions:
            # Extract and flatten all the nested data
            row = {
                # Basic competition info
                'competition_id': competition.competition_id,
                'name': competition.name,
                'starts_at': competition.starts_at,
                'venue': competition.venue,
                
                # Away team details
                'away_team_id': competition.away_team.team_id,
                'away_team_city': competition.away_team.city,
                'away_team_name': competition.away_team.name,
                
                # Home team details
                'home_team_id': competition.home_team.team_id,
                'home_team_city': competition.home_team.city,
                'home_team_name': competition.home_team.name,
                        
                # Availability flags
                'are_starting_lineups_available': competition.are_starting_lineups_available
            }
            
            data.append(row)

        return pd.DataFrame(data)


    def process_draft_group_players(players):
        data = []

        for player in players:
            # Extract and flatten all the nested data
            row = {
                # Basic player info
                'player_id': player.player_id,
                'draftable_id': player.draftable_id,
                'salary': player.salary,
                'position_name': player.position_name,
                'roster_slot_id': player.roster_slot_id,
                'news_status_description': player.news_status_description,
                
                # Name details
                'display_name': player.name_details.display,
                'first_name': player.name_details.first,
                'last_name': player.name_details.last,
                
                # Team details
                'team_abbreviation': player.team_details.abbreviation,
                'team_id': player.team_details.team_id,
                
                # Competition details
                'competition_id': player.competition_details.competition_id,
                'competition_name': player.competition_details.name,
                'starts_at': player.competition_details.starts_at,
                
            }
            
            data.append(row)

        return pd.DataFrame(data)

    contest_games = {}
    contest_players = {}
    for draft_group_id in draft_groups:

        draft_group = Client().draftables(draft_group_id=draft_group_id)

        competitions = process_draft_group_competitions(draft_group.competitions)
        players = process_draft_group_players(draft_group.players)
        contest_games[draft_group_id] = competitions
        contest_players[draft_group_id] = players

    contest_games = pd.concat(contest_games).reset_index().rename({'level_0':'draft_group_id'}, axis=1).drop(['level_1'],axis=1)
    contest_players = pd.concat(contest_players).reset_index().rename({'level_0':'draft_group_id'}, axis=1).drop(['level_1'],axis=1)

    return contest_games, contest_players

def save_to_s3(df, folder_name, aws_bucket='mlbdk', aws_region='us-east-1'):

    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=aws_region)
    
    # Generate S3 key
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    year_folder = datetime.now().year
    s3_key = f"raw_data/draftkings/{folder_name}/{year_folder}/{timestamp}.parquet"
    
    # Upload to S3
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
        df.to_parquet(temp_file.name, index=False)
        s3_client.upload_file(temp_file.name, aws_bucket, s3_key)
        os.unlink(temp_file.name)
        
    return s3_key


def fetch_and_save_daily_slates(rel_date):
    
    contests = fetch_dk_contests(rel_date)
    rel_draft_groups = fetch_dk_draft_groups(contests)
    draft_group_game_type = map_dk_draft_group_to_game_type(rel_draft_groups)

    classic_draft_groups = [k for k, v in draft_group_game_type.items() if v == 28]

    classic_contests = filter_dk_contests(contests, classic_draft_groups)

    contest_games, contest_players = fetch_dk_competitions_and_players(classic_draft_groups)
    

    save_to_s3(classic_contests, 'contests')
    save_to_s3(contest_games, 'games')
    save_to_s3(contest_players, 'contest_players')

    processed_dict = {
        'contests_processed': len(classic_contests),
        'draft_groups_processed': len(classic_draft_groups),
        'games_processed': len(contest_games),
        'players_processed': len(contest_players),
        'date': rel_date
    }

    return processed_dict