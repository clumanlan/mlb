"""
Parses the nested bookmakers/markets/outcomes structure The Odds API returns
for the spreads+totals request (see raw_data/odds/team_odds parquet) into the
run line (favorite side) and total DraftKings has posted for one game.
"""


def extract_game_odds(bookmakers):
    if bookmakers is None or len(bookmakers) == 0:
        return None

    draftkings = next((b for b in bookmakers if b.get("key") == "draftkings"), None)
    if draftkings is None:
        return None

    markets = {m["key"]: m["outcomes"] for m in draftkings.get("markets", [])}

    run_line = None
    spread_outcomes = markets.get("spreads")
    if spread_outcomes is not None and len(spread_outcomes) > 0:
        favorite = min(spread_outcomes, key=lambda o: o["point"])
        run_line = {"team": favorite["name"], "point": favorite["point"], "price": favorite["price"]}

    total = None
    total_outcomes = markets.get("totals")
    if total_outcomes is not None and len(total_outcomes) > 0:
        over = next((o for o in total_outcomes if o["name"] == "Over"), None)
        under = next((o for o in total_outcomes if o["name"] == "Under"), None)
        if over and under:
            total = {"point": over["point"], "over_price": over["price"], "under_price": under["price"]}

    if run_line is None and total is None:
        return None

    return {"run_line": run_line, "total": total}
