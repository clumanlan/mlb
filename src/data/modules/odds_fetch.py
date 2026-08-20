import logging
import requests
import time

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
BOOKMAKERS = "draftkings"
PROP_MARKETS = "pitcher_strikeouts,batter_hits,batter_home_runs,pitcher_hits_allowed"


def api_get(url, params, api_key, retries=3, delay=2):
    params = {**params, "apiKey": api_key}
    for attempt in range(retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logger.info(f"Requests remaining: {response.headers.get('x-requests-remaining')}")
            return response.json()
        elif response.status_code == 429:
            wait = delay * (attempt + 1)
            logger.warning(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)
        else:
            response.raise_for_status()
    raise Exception(f"Failed after {retries} retries")


def get_games(date, api_key):
    url = f"{BASE_URL}/sports/baseball_mlb/odds/"
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "commenceTimeFrom": f"{date}T00:00:00Z",
        "commenceTimeTo": f"{date}T23:59:59Z",
    }
    return [
        {
            "id": g["id"],
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "commence_time": g["commence_time"],
        }
        for g in api_get(url, params, api_key)
    ]


def get_team_odds(date, api_key):
    url = f"{BASE_URL}/sports/baseball_mlb/odds/"
    params = {
        "regions": "us",
        "markets": "spreads,totals",
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "commenceTimeFrom": f"{date}T00:00:00Z",
        "commenceTimeTo": f"{date}T23:59:59Z",
    }
    return api_get(url, params, api_key)


def get_player_props(event_id, api_key):
    url = f"{BASE_URL}/sports/baseball_mlb/events/{event_id}/odds/"
    params = {
        "regions": "us",
        "markets": PROP_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
    }
    return api_get(url, params, api_key)


def get_all_player_props(date, api_key):
    games = get_games(date, api_key)
    logger.info(f"Found {len(games)} games on {date}")

    results = []
    for game in games:
        logger.info(f"Fetching props: {game['away_team']} @ {game['home_team']}")
        props = get_player_props(game["id"], api_key)
        results.append({"game": game, "props": props})
        time.sleep(0.5)

    return results
