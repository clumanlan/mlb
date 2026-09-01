import logging
import requests
import time

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
HISTORICAL_BASE_URL = f"{BASE_URL}/historical"
BOOKMAKERS = "draftkings"
PROP_MARKETS = "pitcher_strikeouts,batter_hits,batter_home_runs,pitcher_hits_allowed"
HISTORICAL_PROP_MARKETS = "pitcher_strikeouts"

_last_requests_used = None


def get_last_usage():
    return _last_requests_used


def api_get(url, params, api_key, retries=3, delay=2):
    global _last_requests_used
    params = {**params, "apiKey": api_key}
    for attempt in range(retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            logger.info(f"Requests remaining: {response.headers.get('x-requests-remaining')}")
            used = response.headers.get("x-requests-used")
            if used is not None:
                _last_requests_used = int(used)
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


def get_historical_games(date, snapshot_date, api_key):
    """List MLB games on `date` as they stood at `snapshot_date` (ISO8601),
    via The Odds API's historical endpoint. Bills at 10x the normal
    markets x regions rate -- use sparingly, for backtests only, never as a
    routine ops-recovery substitute for the live daily pipeline."""
    url = f"{HISTORICAL_BASE_URL}/sports/baseball_mlb/odds/"
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": snapshot_date,
        "commenceTimeFrom": f"{date}T00:00:00Z",
        "commenceTimeTo": f"{date}T23:59:59Z",
    }
    envelope = api_get(url, params, api_key)
    return [
        {
            "id": g["id"],
            "away_team": g["away_team"],
            "home_team": g["home_team"],
            "commence_time": g["commence_time"],
        }
        for g in envelope["data"]
    ]


def get_historical_player_props(event_id, snapshot_date, api_key):
    """Fetch pitcher_strikeouts player-prop odds for one event as they stood
    at `snapshot_date` (ISO8601). Same 10x historical billing rate as
    get_historical_games -- restricted to one market to keep backtest cost
    down."""
    url = f"{HISTORICAL_BASE_URL}/sports/baseball_mlb/events/{event_id}/odds/"
    params = {
        "regions": "us",
        "markets": HISTORICAL_PROP_MARKETS,
        "bookmakers": BOOKMAKERS,
        "oddsFormat": "american",
        "date": snapshot_date,
    }
    envelope = api_get(url, params, api_key)
    return envelope["data"]


def get_all_historical_player_props(date, api_key):
    """Historical counterpart to get_all_player_props: list every game on
    `date`, then fetch pitcher_strikeouts props for each at a snapshot as
    close to that specific game's own commence_time as the API has data for
    -- a single fixed daily snapshot would catch early games after they'd
    already started and late games too far pregame.

    The games LISTING snapshot itself must be early-morning, pregame for
    every game that day (11:00 UTC = 7am ET, safely before any real MLB
    start time) -- an end-of-day snapshot would miss any game whose market
    already closed by then, the same "only returns still-open markets"
    behavior the live /odds endpoint has (see CLAUDE.md's Known Issues)."""
    games = get_historical_games(date, f"{date}T11:00:00Z", api_key)
    logger.info(f"Found {len(games)} historical games on {date}")

    results = []
    for game in games:
        logger.info(f"Fetching historical props: {game['away_team']} @ {game['home_team']}")
        try:
            props = get_historical_player_props(game["id"], game["commence_time"], api_key)
        except Exception as e:
            logger.warning(f"Skipping {game['id']} ({game['away_team']} @ {game['home_team']}): {e}")
            continue
        results.append({"game": game, "props": props})
        time.sleep(0.5)

    return results


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
