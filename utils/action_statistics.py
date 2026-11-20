"""Action normalization keys shared by raw training and evaluation."""

RAW_DATASET_NAME = "uavon_raw_json"


def resolve_action_stats_key(norm_stats, requested=None):
    """Return the requested key, the raw UAV-ON key, or the only available action statistics."""
    available = {
        key for key, value in norm_stats.items()
        if isinstance(value, dict) and "action" in value
    }
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Unknown action statistics key {requested!r}; available: {sorted(available)}")
        return requested
    if RAW_DATASET_NAME in available:
        return RAW_DATASET_NAME
    if len(available) == 1:
        return next(iter(available))
    raise ValueError(f"Set --unnorm_key explicitly; available action statistics: {sorted(available)}")
