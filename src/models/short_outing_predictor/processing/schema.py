# Short outing / bullpen day — a starting pitcher fails to complete a
# normal workload. README's sub-problem menu defines the mechanism as
# "<=4 IP or an explicit opener"; explicit-opener detection needs a
# pre-game "planned opener" flag this baseline doesn't have wired up yet,
# so the baseline label is realized IP alone. An opener's boxscore line is
# itself almost always <=4 IP anyway, so this still captures the great
# majority of the target's real-world cases.
SHORT_OUTING_IP_THRESHOLD = 4.0

__all__ = ['SHORT_OUTING_IP_THRESHOLD']
