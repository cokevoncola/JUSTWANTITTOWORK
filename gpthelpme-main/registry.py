# src/registry.py

# List the public symbols each module should export so imports never drift.
REGISTRY = {
    "smart_mc_models": [
        "SmartMonteCarloPredictor",
        "load_lineups_csv",
        "export_batch_to_csv",
    ],
    "picks": [
        "make_picks",
        "make_picks_smart_mc",
    ],
    # Add more as you grow:
    # "smart_mc_models_heads": ["SmartMonteCarloPredictorHeads"],
    # "smart_csv_processor": ["process_smart_csv"],
}

def debug_exports():
    """
    Returns a list of (module, symbol) that are missing.
    Example: [('picks', 'make_picks_smart_mc')] if that name isn't found.
    """
    import importlib
    missing = []
    for mod, names in REGISTRY.items():
        m = importlib.import_module(f"src.{mod}")
        for n in names:
            if not hasattr(m, n):
                missing.append((mod, n))
    return missing
