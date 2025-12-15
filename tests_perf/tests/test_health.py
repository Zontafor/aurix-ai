def test_health(client, base_url):
    r = client.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data == {"status": "ok"}, data