from game_odds import extract_game_odds


def _dk_bookmaker(markets):
    return [{"key": "draftkings", "title": "DraftKings", "markets": markets}]


class TestExtractGameOdds:
    def test_returns_none_for_empty_bookmakers(self):
        assert extract_game_odds([]) is None

    def test_returns_none_for_none_bookmakers(self):
        assert extract_game_odds(None) is None

    def test_returns_none_when_draftkings_not_present(self):
        bookmakers = [{"key": "fanduel", "title": "FanDuel", "markets": []}]
        assert extract_game_odds(bookmakers) is None

    def test_extracts_run_line_and_total(self):
        bookmakers = _dk_bookmaker([
            {
                "key": "spreads",
                "outcomes": [
                    {"name": "Colorado Rockies", "point": 1.5, "price": -171},
                    {"name": "Washington Nationals", "point": -1.5, "price": 141},
                ],
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 9.5, "price": -118},
                    {"name": "Under", "point": 9.5, "price": -102},
                ],
            },
        ])

        result = extract_game_odds(bookmakers)

        assert result == {
            "run_line": {"team": "Washington Nationals", "point": -1.5, "price": 141},
            "total": {"point": 9.5, "over_price": -118, "under_price": -102},
        }

    def test_run_line_only_when_totals_market_missing(self):
        bookmakers = _dk_bookmaker([
            {
                "key": "spreads",
                "outcomes": [
                    {"name": "Colorado Rockies", "point": 1.5, "price": -171},
                    {"name": "Washington Nationals", "point": -1.5, "price": 141},
                ],
            },
        ])

        result = extract_game_odds(bookmakers)

        assert result == {
            "run_line": {"team": "Washington Nationals", "point": -1.5, "price": 141},
            "total": None,
        }

    def test_total_only_when_spreads_market_missing(self):
        bookmakers = _dk_bookmaker([
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 9.5, "price": -118},
                    {"name": "Under", "point": 9.5, "price": -102},
                ],
            },
        ])

        result = extract_game_odds(bookmakers)

        assert result == {
            "run_line": None,
            "total": {"point": 9.5, "over_price": -118, "under_price": -102},
        }

    def test_returns_none_when_draftkings_has_no_markets(self):
        bookmakers = _dk_bookmaker([])
        assert extract_game_odds(bookmakers) is None
