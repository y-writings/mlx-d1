from mlx_d1.bench import measure, percentile, summarize


def test_percentile_and_summary():
    samples = [10, 20, 30, 40, 100]
    assert percentile(samples, 0.50) == 30
    assert percentile(samples, 0.95) == 88
    stats = summarize([float(value) for value in samples])
    assert stats["count"] == 5
    assert stats["mean_ms"] == 40
    assert stats["min_ms"] == 10
    assert stats["max_ms"] == 100
    assert stats["p50_ms"] == 30
    assert stats["p95_ms"] == 88


def test_measure_excludes_warmup_calls():
    seen = []

    def post(payload):
        seen.append(payload["n"])

    stats = measure(post, {"n": 1}, requests=3, warmup=2)
    assert seen == [1, 1, 1, 1, 1]
    assert stats["count"] == 3
