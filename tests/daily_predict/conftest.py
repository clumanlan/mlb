import os
import sys

# Deliberately does NOT add src/lambdas/daily_predict to sys.path — that
# directory contains handler.py, the same generic filename every other
# Lambda uses, and a permanent (session-lifetime) conftest.py sys.path
# insertion would silently shadow other Lambdas' handler.py for any test
# that does a bare `from handler import handler` (daily_odds_fetch's tests
# still do — see CLAUDE.md's Test Coverage section). predict.py is loaded
# directly by file path in test_predict.py instead, the same isolated
# pattern the *_handler.py tests already use for this exact reason.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src/features/transforms"))
