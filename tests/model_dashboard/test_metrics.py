from sklearn.metrics import log_loss as sklearn_log_loss

from shared.model_dashboard.logic.metrics import per_row_loss


def test_per_row_loss_matches_sklearn(toy_pa_data):
    for _, row in toy_pa_data.iterrows():
        expected = sklearn_log_loss([row["is_hit"]], [row["pred_prob"]], labels=[0, 1])
        actual = per_row_loss(row["is_hit"], row["pred_prob"])
        assert abs(actual - expected) < 1e-10, (
            f"y={row['is_hit']}, p={row['pred_prob']}: got {actual}, expected {expected}"
        )
