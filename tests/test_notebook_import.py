def test_notebook_imports_and_app_exists():
    import pytest
    pytest.importorskip("marimo")
    # Import the notebook module from a known path (root or notebooks/)
    import importlib.util
    from pathlib import Path

    candidate_paths = [
        Path("portfolio_growth_simulator.py"),
        Path("notebooks/portfolio_growth_simulator.py"),
    ]
    nb = None
    for p in candidate_paths:
        if p.exists():
            spec = importlib.util.spec_from_file_location("nb_under_test", p)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            nb = mod
            break
    assert nb is not None, "Could not locate portfolio_growth_simulator.py"
    assert hasattr(nb, "app"), "Notebook must expose a marimo App as `app`"
