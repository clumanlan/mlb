# Residual error analysis v2 — batters_faced_predictor tuned XGBoost (v2 features)

Val season 2024, MAE 2.6471 (sanity check vs. v2_results.md's 2.6471)

Cold-start (0-2 starts) is already investigated (v3, flat result, genuine
small-sample-variance ceiling) — this pass excludes it from the worked
examples/buckets below to look for a DIFFERENT lever.

Cold-start MAE 3.1205 (n=970) vs. established MAE 2.5288 (n=3,820), shown for reference only.

## Top 25 under-predictions, excluding cold-start

| player_name      |   personId |   gamepk |   realized_batters_faced |   predicted |   residual |   expected_batters_faced |   expected_batters_faced_weight |   pitcher_this_season_start_pa_starts_n |   pitcher_last3_start_pa_avg_pa_per_start |   pa_trend_direction |   pitcher_days_since_last_start |   pitcher_workload_density |   ip |   h |   r |   er |   bb |   k |   hr |   p |   s | outcome   | decision    |   is_home | opp_team_name        | venue_name               |   score_margin |   is_doubleheader_g2 | weather_condition   |   weather_temp |   game_duration_minutes |   month | day_of_week   |
|:-----------------|-----------:|---------:|-------------------------:|------------:|-----------:|-------------------------:|--------------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|--------------------------------:|---------------------------:|-----:|----:|----:|-----:|-----:|----:|-----:|----:|----:|:----------|:------------|----------:|:---------------------|:-------------------------|---------------:|---------------------:|:--------------------|---------------:|------------------------:|--------:|:--------------|
| Chris Flexen     |     623167 |   746755 |                       28 |       16.49 |      11.51 |                    21.67 |                            0.83 |                                      24 |                                     21.33 |                   -1 |                               4 |                      24.5  |  6.1 |   9 |   3 |    3 |    2 |   4 |    0 |  95 |  61 | L         | no_decision |         1 | Texas Rangers        | Guaranteed Rate Field    |              2 |                    0 | Cloudy              |             77 |                     151 |       8 | Wednesday     |
| Frankie Montas   |     593423 |   746718 |                       24 |       12.86 |      11.14 |                    19.2  |                            0.5  |                                       5 |                                     13    |                   -1 |                              16 |                       1    |  6   |   4 |   2 |    1 |    1 |   7 |    0 |  87 |  61 | L         | no_decision |         1 | Arizona Diamondbacks | Great American Ball Park |              4 |                    0 | Cloudy              |             75 |                     147 |       5 | Tuesday       |
| Jonathan Cannon  |     686563 |   746783 |                       34 |       23.03 |      10.97 |                    22    |                            0.44 |                                       4 |                                     23.33 |                    1 |                               6 |                      17.33 |  8.2 |   7 |   0 |    0 |    1 |   4 |    0 | 106 |  70 | W         | no_decision |         1 | Houston Astros       | Guaranteed Rate Field    |              2 |                    0 | Partly Cloudy       |             88 |                     122 |       6 | Wednesday     |
| Yariel Rodríguez |     684320 |   744914 |                       23 |       12.05 |      10.95 |                    17.94 |                            0.55 |                                       6 |                                     11    |                   -1 |                               5 |                       3    |  6.2 |   2 |   1 |    1 |    2 |   6 |    1 |  83 |  51 | L         | no_decision |         1 | Houston Astros       | Rogers Centre            |              2 |                    0 | Sunny               |             69 |                     136 |       7 | Monday        |
| Emerson Hancock  |     676106 |   745245 |                       29 |       18.38 |      10.62 |                    18.79 |                            0.55 |                                       6 |                                     19.67 |                   -1 |                              37 |                       2.05 |  7   |   6 |   2 |    2 |    2 |   1 |    2 |  92 |  64 | <NA>      | no_decision |         1 | Chicago White Sox    | T-Mobile Park            |              1 |                    0 | Partly Cloudy       |             67 |                     148 |       6 | Friday        |
| Cooper Criswell  |     681867 |   746936 |                       29 |       18.75 |      10.25 |                    19.12 |                            0.69 |                                      11 |                                     18.67 |                   -1 |                              27 |                       3.26 |  6   |   9 |   5 |    5 |    2 |   1 |    1 | 104 |  70 | L         | no_decision |         1 | Kansas City Royals   | Fenway Park              |              5 |                    0 | Partly Cloudy       |             82 |                     136 |       7 | Friday        |
| Justin Wrobleski |     680736 |   747157 |                       29 |       19.41 |       9.59 |                    20.3  |                            0.5  |                                       5 |                                     19.67 |                   -1 |                              16 |                       4.94 |  5.1 |  10 |  10 |   10 |    2 |   2 |    2 |  99 |  58 | L         | no_decision |         0 | Arizona Diamondbacks | Chase Field              |             11 |                    0 | Roof Closed         |             78 |                     158 |       9 | Sunday        |
| Kevin Gausman    |     592332 |   744908 |                       34 |       24.48 |       9.52 |                    23.76 |                            0.8  |                                      20 |                                     27.33 |                    1 |                               6 |                      16.17 |  9   |   4 |   3 |    3 |    3 |   8 |    0 | 118 |  81 | W         | no_decision |         1 | Texas Rangers        | Rogers Centre            |              4 |                    0 | Sunny               |             76 |                     145 |       7 | Saturday      |
| Hunter Brown     |     686613 |   745885 |                       32 |       22.71 |       9.29 |                    21.76 |                            0.76 |                                      16 |                                     23.33 |                    1 |                               5 |                      19.8  |  6   |  12 |   7 |    7 |    1 |   6 |    1 | 105 |  69 | L         | no_decision |         0 | Minnesota Twins      | Target Field             |              6 |                    0 | Partly Cloudy       |             78 |                     157 |       7 | Saturday      |
| Erick Fedde      |     607200 |   746805 |                       32 |       22.83 |       9.17 |                    21.89 |                            0.44 |                                       4 |                                     22.67 |                    1 |                               5 |                      19    |  8.1 |   7 |   2 |    2 |    0 |   9 |    1 | 108 |  72 | W         | no_decision |         1 | Tampa Bay Rays       | Guaranteed Rate Field    |              2 |                    0 | Cloudy              |             69 |                     126 |       4 | Sunday        |
| Ranger Suarez    |     624133 |   745593 |                       32 |       22.91 |       9.09 |                    23.19 |                            0.38 |                                       3 |                                     21    |                    0 |                               5 |                      19.4  |  9   |   7 |   0 |    0 |    1 |   8 |    0 | 112 |  79 | W         | no_decision |         1 | Colorado Rockies     | Citizens Bank Park       |              5 |                    0 | Clear               |             72 |                     127 |       4 | Tuesday       |
| Logan Webb       |     657277 |   745306 |                       34 |       25.04 |       8.96 |                    25.64 |                            0.81 |                                      22 |                                     26    |                    1 |                               6 |                      14.83 |  9   |   5 |   0 |    0 |    1 |   6 |    0 | 106 |  73 | W         | no_decision |         1 | Oakland Athletics    | Oracle Park              |              1 |                    0 | Partly Cloudy       |             62 |                     115 |       8 | Thursday      |
| Kevin Gausman    |     592332 |   745652 |                       32 |       23.05 |       8.95 |                    22.23 |                            0.71 |                                      12 |                                     23.33 |                    1 |                               5 |                      19.4  |  9   |   5 |   0 |    0 |    1 |  10 |    0 | 109 |  76 | W         | no_decision |         0 | Oakland Athletics    | Oakland Coliseum         |              7 |                    0 | Cloudy              |             65 |                     147 |       6 | Saturday      |
| Roddery Muñoz    |     682610 |   746367 |                       30 |       21.38 |       8.62 |                    21.65 |                            0.58 |                                       7 |                                     23.67 |                    1 |                              12 |                       6.58 |  6.2 |   5 |   6 |    4 |    2 |   2 |    0 |  94 |  58 | L         | no_decision |         0 | Houston Astros       | Minute Maid Park         |              3 |                    0 | Roof Closed         |             73 |                     154 |       7 | Friday        |
| Nathan Eovaldi   |     543135 |   745975 |                       31 |       22.59 |       8.41 |                    22.74 |                            0.71 |                                      12 |                                     22.33 |                   -1 |                               5 |                      19.6  |  7   |   9 |   5 |    5 |    1 |   4 |    2 | 101 |  75 | <NA>      | no_decision |         0 | Milwaukee Brewers    | American Family Field    |              1 |                    0 | Partly Cloudy       |             83 |                     163 |       6 | Wednesday     |
| Colin Rea        |     607067 |   745932 |                       30 |       21.59 |       8.41 |                    22.91 |                            0.83 |                                      25 |                                     20    |                   -1 |                               9 |                       7.44 |  5.2 |  10 |   5 |    5 |    3 |   5 |    1 | 106 |  64 | L         | no_decision |         1 | New York Mets        | American Family Field    |              5 |                    0 | Sunny               |             72 |                     154 |       9 | Sunday        |
| Bailey Falter    |     663559 |   746114 |                       22 |       13.62 |       8.38 |                    20.37 |                            0.79 |                                      19 |                                     13.33 |                   -1 |                               5 |                       4    |  4   |   7 |   4 |    4 |    3 |   2 |    0 |  93 |  60 | <NA>      | no_decision |         0 | Los Angeles Dodgers  | Dodger Stadium           |              1 |                    0 | Sunny               |             88 |                     204 |       8 | Sunday        |
| Aaron Nola       |     605400 |   745424 |                       32 |       23.87 |       8.13 |                    24.6  |                            0.5  |                                       5 |                                     26.33 |                    1 |                               5 |                      18.2  |  8   |   7 |   3 |    3 |    1 |  10 |    1 | 106 |  74 | W         | no_decision |         0 | San Diego Padres     | Petco Park               |              6 |                    0 | Cloudy              |             61 |                     162 |       4 | Saturday      |
| Tyler Alexander  |     641302 |   745092 |                       29 |       20.92 |       8.08 |                    14.56 |                            0.44 |                                       4 |                                     19.33 |                   -1 |                               6 |                      12.5  |  7   |   7 |   6 |    6 |    2 |   3 |    3 |  90 |  64 | L         | no_decision |         1 | New York Yankees     | Tropicana Field          |              4 |                    0 | Dome                |             72 |                     176 |       5 | Sunday        |
| Tarik Skubal     |     669373 |   746612 |                       31 |       22.97 |       8.03 |                    23.43 |                            0.78 |                                      18 |                                     23.33 |                   -1 |                              10 |                      10.1  |  7   |  10 |   1 |    1 |    1 |   6 |    0 |  97 |  73 | W         | no_decision |         0 | Cleveland Guardians  | Progressive Field        |              6 |                    0 | Cloudy              |             76 |                     158 |       7 | Monday        |
| Luis Severino    |     622663 |   745785 |                       32 |       23.99 |       8.01 |                    23.85 |                            0.82 |                                      23 |                                     20.67 |                   -1 |                               6 |                      15.33 |  9   |   4 |   0 |    0 |    1 |   8 |    0 | 113 |  78 | W         | no_decision |         1 | Miami Marlins        | Citi Field               |              4 |                    0 | Cloudy              |             79 |                     146 |       8 | Saturday      |
| Tyler Glasnow    |     607192 |   746154 |                       31 |       23    |       8    |                    21.95 |                            0.5  |                                       5 |                                     24.33 |                    1 |                               6 |                      15.67 |  8   |   7 |   0 |    0 |    0 |  10 |    0 | 101 |  70 | W         | no_decision |         1 | New York Mets        | Dodger Stadium           |             10 |                    0 | Sunny               |             75 |                     138 |       4 | Sunday        |
| Nick Pivetta     |     601713 |   744918 |                       30 |       22.08 |       7.92 |                    21.33 |                            0.64 |                                       9 |                                     22    |                    1 |                               5 |                      18.6  |  7   |   9 |   3 |    3 |    1 |   4 |    2 | 109 |  73 | W         | no_decision |         0 | Toronto Blue Jays    | Rogers Centre            |              4 |                    0 | Clear               |             75 |                     163 |       6 | Monday        |
| Clarke Schmidt   |     657376 |   745907 |                       30 |       22.09 |       7.91 |                    21.47 |                            0.58 |                                       7 |                                     22    |                    0 |                               6 |                      14.5  |  8   |   3 |   0 |    0 |    0 |   8 |    0 | 103 |  69 | W         | no_decision |         0 | Minnesota Twins      | Target Field             |              5 |                    0 | Partly Cloudy       |             60 |                     135 |       5 | Thursday      |
| Frankie Montas   |     593423 |   746689 |                       30 |       22.24 |       7.76 |                    21.14 |                            0.76 |                                      16 |                                     23    |                    1 |                               6 |                      15.67 |  7   |   8 |   5 |    5 |    2 |   7 |    2 | 101 |  65 | L         | no_decision |         1 | Colorado Rockies     | Great American Ball Park |              1 |                    0 | Clear               |             79 |                     147 |       7 | Wednesday     |

## Top 25 over-predictions, excluding cold-start

| player_name        |   personId |   gamepk |   realized_batters_faced |   predicted |   residual |   expected_batters_faced |   expected_batters_faced_weight |   pitcher_this_season_start_pa_starts_n |   pitcher_last3_start_pa_avg_pa_per_start |   pa_trend_direction |   pitcher_days_since_last_start |   pitcher_workload_density |   ip |   h |   r |   er |   bb |   k |   hr |   p |   s | outcome   | decision    |   is_home | opp_team_name        | venue_name               |   score_margin |   is_doubleheader_g2 | weather_condition   |   weather_temp |   game_duration_minutes |   month | day_of_week   |
|:-------------------|-----------:|---------:|-------------------------:|------------:|-----------:|-------------------------:|--------------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|--------------------------------:|---------------------------:|-----:|----:|----:|-----:|-----:|----:|-----:|----:|----:|:----------|:------------|----------:|:---------------------|:-------------------------|---------------:|---------------------:|:--------------------|---------------:|------------------------:|--------:|:--------------|
| Zac Gallen         |     668678 |   745819 |                        1 |       23.93 |     -22.93 |                    23.61 |                            0.67 |                                      10 |                                     25.67 |                    1 |                               6 |                      16    |  0   |   1 |   0 |    0 |    0 |   0 |    0 |   6 |   4 | <NA>      | no_decision |         0 | New York Mets        | Citi Field               |              1 |                    0 | Partly Cloudy       |             69 |                     163 |       5 | Thursday      |
| Dylan Cease        |     656302 |   745467 |                        4 |       23.75 |     -19.75 |                    23.37 |                            0.82 |                                      23 |                                     24    |                    1 |                               6 |                      16.83 |  1   |   1 |   0 |    0 |    0 |   2 |    0 |  14 |  10 | <NA>      | no_decision |         0 | Pittsburgh Pirates   | PNC Park                 |              6 |                    0 | Cloudy              |             77 |                     157 |       8 | Tuesday       |
| James Paxton       |     572020 |   746923 |                        3 |       22.28 |     -19.28 |                    21.77 |                            0.8  |                                      20 |                                     22.67 |                    1 |                               6 |                      15    |  0.2 |   2 |   0 |    0 |    0 |   0 |    0 |   5 |   4 | <NA>      | no_decision |         1 | Houston Astros       | Fenway Park              |              8 |                    0 | Partly Cloudy       |             81 |                     167 |       8 | Sunday        |
| Ryan Feltner       |     663372 |   746521 |                        5 |       23.76 |     -18.76 |                    23.39 |                            0.81 |                                      22 |                                     26.33 |                    1 |                               6 |                      15.5  |  1   |   2 |   1 |    1 |    0 |   1 |    0 |  24 |  15 | <NA>      | no_decision |         1 | New York Mets        | Coors Field              |              2 |                    0 | Partly Cloudy       |             73 |                     166 |       8 | Thursday      |
| Ranger Suarez      |     624133 |   745575 |                        6 |       24.74 |     -18.74 |                    24.59 |                            0.69 |                                      11 |                                     25    |                    1 |                               6 |                      16.67 |  2   |   0 |   0 |    0 |    0 |   2 |    0 |  23 |  16 | <NA>      | no_decision |         1 | St. Louis Cardinals  | Citizens Bank Park       |              5 |                    0 | Clear               |             81 |                     148 |       6 | Saturday      |
| Joe Ross           |     605452 |   746066 |                        4 |       22.61 |     -18.61 |                    22.47 |                            0.62 |                                       8 |                                     21    |                   -1 |                               6 |                      15.17 |  1   |   1 |   0 |    0 |    1 |   0 |    0 |  15 |   9 | <NA>      | no_decision |         0 | Miami Marlins        | loanDepot park           |              1 |                    0 | Roof Closed         |             72 |                     169 |       5 | Monday        |
| Frankie Montas     |     593423 |   746727 |                        3 |       21.51 |     -18.51 |                    21    |                            0.44 |                                       4 |                                     20.33 |                   -1 |                               6 |                      11    |  0.2 |   0 |   0 |    0 |    1 |   0 |    0 |  16 |  11 | <NA>      | no_decision |         1 | Los Angeles Angels   | Great American Ball Park |              3 |                    0 | Cloudy              |             49 |                     156 |       4 | Sunday        |
| Alek Manoah        |     666201 |   746790 |                        6 |       23.6  |     -17.6  |                    22.99 |                            0.44 |                                       4 |                                     24.67 |                    1 |                               5 |                      19.4  |  1.2 |   1 |   0 |    0 |    0 |   3 |    0 |  24 |  15 | <NA>      | no_decision |         0 | Chicago White Sox    | Guaranteed Rate Field    |              2 |                    0 | Clear               |             60 |                     151 |       5 | Wednesday     |
| Reynaldo López     |     625643 |   744807 |                        5 |       22.43 |     -17.43 |                    22.32 |                            0.81 |                                      22 |                                     23    |                    1 |                               5 |                      18.2  |  1   |   1 |   0 |    0 |    0 |   1 |    0 |  25 |  16 | <NA>      | no_decision |         0 | Washington Nationals | Nationals Park           |             12 |                    0 | Clear               |             82 |                     155 |       9 | Tuesday       |
| Kutter Crawford    |     676710 |   746942 |                        6 |       23.36 |     -17.36 |                    22.71 |                            0.76 |                                      16 |                                     24.67 |                    1 |                               5 |                      16.6  |  1.1 |   0 |   0 |    0 |    2 |   2 |    0 |  23 |  14 | <NA>      | no_decision |         1 | Toronto Blue Jays    | Fenway Park              |              3 |                    0 | Cloudy              |             81 |                     156 |       6 | Wednesday     |
| Joe Ryan           |     657746 |   746842 |                        7 |       24.11 |     -17.11 |                    23.86 |                            0.81 |                                      22 |                                     24    |                   -1 |                               5 |                      20.6  |  2   |   1 |   1 |    1 |    0 |   2 |    1 |  33 |  22 | <NA>      | no_decision |         0 | Chicago Cubs         | Wrigley Field            |              6 |                    0 | Sunny               |             75 |                     164 |       8 | Wednesday     |
| Garrett Crochet    |     676979 |   746774 |                        6 |       22.9  |     -16.9  |                    22.04 |                            0.79 |                                      19 |                                     22.67 |                    1 |                               6 |                      15.5  |  2   |   0 |   0 |    0 |    0 |   4 |    0 |  28 |  22 | <NA>      | no_decision |         1 | Pittsburgh Pirates   | Guaranteed Rate Field    |              3 |                    0 | Clear               |             78 |                     131 |       7 | Saturday      |
| Justin Steele      |     657006 |   746841 |                        8 |       24.72 |     -16.72 |                    24.52 |                            0.79 |                                      19 |                                     26    |                    1 |                               7 |                      14.43 |  2   |   0 |   0 |    0 |    1 |   3 |    0 |  36 |  23 | <NA>      | no_decision |         1 | Toronto Blue Jays    | Wrigley Field            |              1 |                    0 | Cloudy              |             78 |                     151 |       8 | Saturday      |
| Paul Skenes        |     694973 |   745685 |                        6 |       22.29 |     -16.29 |                    22.82 |                            0.81 |                                      21 |                                     21.33 |                   -1 |                               6 |                      12.17 |  2   |   0 |   0 |    0 |    0 |   3 |    0 |  23 |  17 | <NA>      | no_decision |         0 | New York Yankees     | Yankee Stadium           |              5 |                    0 | Overcast            |             65 |                     169 |       9 | Saturday      |
| Seth Lugo          |     607625 |   747066 |                        8 |       24.14 |     -16.14 |                    25.38 |                            0.86 |                                      32 |                                     24.33 |                   -1 |                               6 |                      14.83 |  2   |   1 |   0 |    0 |    1 |   3 |    0 |  36 |  23 | <NA>      | no_decision |         0 | Atlanta Braves       | Truist Park              |              1 |                    0 | Cloudy              |             72 |                     140 |       9 | Saturday      |
| Mitchell Parker    |     680730 |   745963 |                        7 |       22.97 |     -15.97 |                    23.13 |                            0.75 |                                      15 |                                     24.33 |                    1 |                               5 |                      18.2  |  0.2 |   3 |   5 |    5 |    2 |   2 |    0 |  46 |  28 | <NA>      | no_decision |         0 | Milwaukee Brewers    | American Family Field    |              1 |                    0 | Partly Cloudy       |             86 |                     180 |       7 | Saturday      |
| Michael Lorenzen   |     547179 |   746594 |                        7 |       22.63 |     -15.63 |                    22.78 |                            0.81 |                                      22 |                                     23    |                    1 |                               6 |                      15.83 |  1.2 |   1 |   0 |    0 |    1 |   1 |    0 |  28 |  14 | <NA>      | no_decision |         0 | Cleveland Guardians  | Progressive Field        |              5 |                    0 | Clear               |             89 |                     161 |       8 | Tuesday       |
| Ben Lively         |     594902 |   746747 |                        7 |       22.6  |     -15.6  |                    22.81 |                            0.83 |                                      25 |                                     23.67 |                    1 |                               6 |                      13    |  2   |   1 |   0 |    0 |    0 |   2 |    0 |  28 |  20 | <NA>      | no_decision |         0 | Chicago White Sox    | Guaranteed Rate Field    |              5 |                    0 | Clear               |             80 |                     157 |       9 | Tuesday       |
| Reese Olson        |     681857 |   744910 |                        7 |       22.39 |     -15.39 |                    22.5  |                            0.78 |                                      18 |                                     23.33 |                    1 |                              10 |                       9.5  |  2   |   1 |   0 |    0 |    2 |   0 |    0 |  30 |  16 | <NA>      | no_decision |         0 | Toronto Blue Jays    | Rogers Centre            |              4 |                    0 | Partly Cloudy       |             75 |                     156 |       7 | Saturday      |
| Gavin Williams     |     668909 |   746099 |                        7 |       22.3  |     -15.3  |                    21.45 |                            0.71 |                                      12 |                                     21.33 |                   -1 |                               5 |                      19.6  |  0.2 |   2 |   5 |    5 |    3 |   0 |    0 |  37 |  17 | L         | no_decision |         0 | Los Angeles Dodgers  | Dodger Stadium           |              5 |                    0 | Partly Cloudy       |             92 |                     154 |       9 | Sunday        |
| Yoshinobu Yamamoto |     808967 |   746135 |                        8 |       23.26 |     -15.26 |                    21.94 |                            0.72 |                                      13 |                                     24.67 |                    1 |                               8 |                      13.25 |  2   |   1 |   0 |    0 |    1 |   1 |    0 |  28 |  14 | <NA>      | no_decision |         1 | Kansas City Royals   | Dodger Stadium           |              5 |                    0 | Clear               |             77 |                     152 |       6 | Sunday        |
| Yariel Rodríguez   |     684320 |   746942 |                        4 |       19.2  |     -15.2  |                    19.34 |                            0.5  |                                       5 |                                     16.33 |                    1 |                               5 |                      10.4  |  1   |   0 |   0 |    0 |    1 |   0 |    0 |  15 |   8 | <NA>      | no_decision |         0 | Boston Red Sox       | Fenway Park              |              3 |                    0 | Cloudy              |             81 |                     156 |       6 | Wednesday     |
| Joe Boyle          |     671212 |   745667 |                        7 |       21.84 |     -14.84 |                    20.18 |                            0.55 |                                       6 |                                     20.33 |                    0 |                               6 |                      15.17 |  1   |   1 |   4 |    4 |    3 |   1 |    1 |  35 |  16 | L         | no_decision |         1 | Miami Marlins        | Oakland Coliseum         |              9 |                    0 | Partly Cloudy       |             59 |                     172 |       5 | Sunday        |
| Ryne Nelson        |     669194 |   745356 |                        7 |       21.8  |     -14.8  |                    21.19 |                            0.38 |                                       3 |                                     21.33 |                    0 |                               5 |                      16.2  |  2   |   2 |   0 |    0 |    0 |   0 |    0 |  27 |  17 | <NA>      | no_decision |         0 | San Francisco Giants | Oracle Park              |              5 |                    0 | Partly Cloudy       |             57 |                     132 |       4 | Friday        |
| Clayton Kershaw    |     477132 |   747156 |                        7 |       21.78 |     -14.78 |                    21.68 |                            0.55 |                                       6 |                                     22.33 |                    1 |                               6 |                      14.67 |  1   |   3 |   3 |    3 |    1 |   0 |    1 |  27 |  17 | <NA>      | no_decision |         0 | Arizona Diamondbacks | Chase Field              |              1 |                    0 | Roof Closed         |             78 |                     193 |       8 | Saturday      |

## Bucket: weather_condition

| weather_condition   |   mean_residual (bias) |   MAE |    n |
|:--------------------|-----------------------:|------:|-----:|
| Unknown             |                  3.876 | 3.876 |    2 |
| Drizzle             |                 -1.026 | 2.952 |   17 |
| Rain                |                 -1.941 | 2.925 |   17 |
| Cloudy              |                 -0.031 | 2.829 |  534 |
| Clear               |                  0.173 | 2.547 |  796 |
| Sunny               |                 -0.238 | 2.503 |  435 |
| Partly Cloudy       |                  0.174 | 2.461 | 1129 |
| Roof Closed         |                  0.03  | 2.441 |  594 |
| Dome                |                  0.04  | 2.403 |  128 |
| Overcast            |                 -0.144 | 2.315 |  168 |

## Bucket: month

|   month |   mean_residual (bias) |   MAE |   n |
|--------:|-----------------------:|------:|----:|
|       8 |                  0.12  | 2.613 | 738 |
|       7 |                  0.205 | 2.579 | 627 |
|       6 |                  0.016 | 2.557 | 721 |
|       9 |                 -0.449 | 2.53  | 707 |
|       4 |                  0.305 | 2.459 | 330 |
|       5 |                  0.229 | 2.397 | 697 |

## Bucket: day_of_week

| day_of_week   |   mean_residual (bias) |   MAE |   n |
|:--------------|-----------------------:|------:|----:|
| Monday        |                  0.521 | 2.682 | 238 |
| Wednesday     |                  0.005 | 2.618 | 686 |
| Saturday      |                 -0.108 | 2.564 | 813 |
| Friday        |                  0.452 | 2.544 | 400 |
| Sunday        |                 -0.116 | 2.533 | 738 |
| Tuesday       |                  0.053 | 2.507 | 509 |
| Thursday      |                  0.02  | 2.242 | 436 |

## Bucket: is_doubleheader_g2

|   is_doubleheader_g2 |   mean_residual (bias) |   MAE |    n |
|---------------------:|-----------------------:|------:|-----:|
|                    1 |                  0.596 | 3.472 |   14 |
|                    0 |                  0.043 | 2.525 | 3806 |

## Bucket: decision

| decision    |   mean_residual (bias) |   MAE |    n |
|:------------|-----------------------:|------:|-----:|
| no_decision |                  0.045 | 2.529 | 3820 |

## Bucket: venue_name

| venue_name                  |   mean_residual (bias) |   MAE |   n |
|:----------------------------|-----------------------:|------:|----:|
| Rickwood Field              |                 -2.239 | 5.659 |   2 |
| London Stadium              |                 -1.119 | 4.594 |   4 |
| Great American Ball Park    |                 -0.304 | 3.163 | 118 |
| Fenway Park                 |                  0.004 | 3.005 | 131 |
| Guaranteed Rate Field       |                 -0.352 | 2.957 | 120 |
| loanDepot park              |                 -0.158 | 2.901 | 117 |
| Citizens Bank Park          |                  0.238 | 2.856 | 123 |
| PNC Park                    |                  0.001 | 2.801 | 132 |
| Oakland Coliseum            |                  0.662 | 2.759 | 116 |
| Coors Field                 |                  0.335 | 2.713 | 126 |
| Rogers Centre               |                  0.556 | 2.667 | 137 |
| Wrigley Field               |                 -0.156 | 2.658 | 131 |
| Dodger Stadium              |                  0.046 | 2.573 | 115 |
| Comerica Park               |                  0.16  | 2.531 | 120 |
| Target Field                |                 -0.111 | 2.51  | 130 |
| Nationals Park              |                 -0.064 | 2.507 | 137 |
| Chase Field                 |                 -0.045 | 2.504 | 125 |
| Progressive Field           |                 -0.259 | 2.493 | 130 |
| Truist Park                 |                  0.313 | 2.479 | 129 |
| Citi Field                  |                 -0.206 | 2.439 | 124 |
| Tropicana Field             |                  0.009 | 2.41  | 126 |
| Kauffman Stadium            |                 -0.066 | 2.384 | 129 |
| American Family Field       |                 -0.122 | 2.383 | 137 |
| Oracle Park                 |                 -0.224 | 2.362 | 125 |
| Angel Stadium               |                  0.627 | 2.35  | 130 |
| Minute Maid Park            |                  0.403 | 2.329 | 128 |
| Oriole Park at Camden Yards |                  0.219 | 2.328 | 129 |
| T-Mobile Park               |                  0.053 | 2.276 | 132 |
| Yankee Stadium              |                 -0.454 | 2.21  | 130 |
| Petco Park                  |                  0.274 | 2.193 | 130 |
| Globe Life Field            |                 -0.049 | 2.166 | 125 |
| Busch Stadium               |                  0.012 | 2.012 | 127 |
| Estadio Alfredo Harp Helu   |                  1.885 | 1.885 |   3 |
| Journey Bank Ballpark       |                  0.37  | 1.29  |   2 |

## Bucket: score_margin_bucket

| score_margin_bucket   |   mean_residual (bias) |   MAE |    n |
|:----------------------|-----------------------:|------:|-----:|
| 7+ (blowout)          |                 -0.535 | 2.839 |  509 |
| 4-6                   |                  0.177 | 2.722 | 1034 |
| <=1 (close)           |                  0.2   | 2.389 | 1060 |
| 2-3                   |                  0.039 | 2.357 | 1217 |

## Bucket: opp_onbase_bucket

| opp_onbase_bucket     |   mean_residual (bias) |   MAE |   n |
|:----------------------|-----------------------:|------:|----:|
| Q1 (weakest lineup)   |                 -0.038 | 2.604 | 955 |
| Q2                    |                  0.031 | 2.559 | 955 |
| Q3                    |                  0.101 | 2.492 | 955 |
| Q4 (strongest lineup) |                  0.084 | 2.461 | 955 |

## Bucket: rest_days_bucket

| rest_days_bucket   |   mean_residual (bias) |   MAE |    n |
|:-------------------|-----------------------:|------:|-----:|
| <=4 (short rest)   |                  0.2   | 2.692 |   10 |
| 7+ (extra rest)    |                  0.161 | 2.626 |  779 |
| 5 (normal)         |                 -0.101 | 2.582 | 1298 |
| 6                  |                  0.106 | 2.449 | 1721 |

## Bucket: workload_density_bucket

| workload_density_bucket   |   mean_residual (bias) |   MAE |   n |
|:--------------------------|-----------------------:|------:|----:|
| Q1 (lightest recent load) |                  0.07  | 2.624 | 983 |
| Q4 (heaviest recent load) |                 -0.084 | 2.544 | 953 |
| Q2                        |                  0.1   | 2.523 | 984 |
| Q3                        |                  0.093 | 2.415 | 900 |

## Bucket: pa_trend_direction

|   pa_trend_direction |   mean_residual (bias) |   MAE |    n |
|---------------------:|-----------------------:|------:|-----:|
|                    1 |                 -0.041 | 2.553 | 1998 |
|                   -1 |                  0.188 | 2.511 | 1508 |
|                    0 |                 -0.1   | 2.458 |  314 |

