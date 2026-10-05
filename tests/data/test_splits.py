from semcom.data.splits import train_val_indices, train_val_test_indices


def test_train_val_split_sizes_and_disjoint():
    train_idx, val_idx = train_val_indices(n=1000, val_fraction=0.2, seed=42)
    assert len(val_idx) == 200
    assert len(train_idx) == 800
    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))


def test_train_val_split_is_deterministic_for_same_seed():
    a_train, a_val = train_val_indices(n=500, val_fraction=0.1, seed=7)
    b_train, b_val = train_val_indices(n=500, val_fraction=0.1, seed=7)
    assert (a_train == b_train).all()
    assert (a_val == b_val).all()


def test_train_val_split_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "splits.json"
    a_train, a_val = train_val_indices(n=200, val_fraction=0.1, seed=1, cache_path=cache_path)
    assert cache_path.exists()
    b_train, b_val = train_val_indices(n=200, val_fraction=0.1, seed=999, cache_path=cache_path)
    assert (a_train == b_train).all()
    assert (a_val == b_val).all()


def test_train_val_test_split_sizes_and_disjoint():
    train_idx, val_idx, test_idx = train_val_test_indices(n=1000, val_fraction=0.1, test_fraction=0.1, seed=42)
    assert len(val_idx) == 100
    assert len(test_idx) == 100
    assert len(train_idx) == 800
    all_ids = train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
    assert len(set(all_ids)) == 1000
