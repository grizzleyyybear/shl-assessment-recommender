import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.catalog import load_catalog
from app.guardrails import validate_recommendations


def test_catalog_loads_and_has_items():
    catalog = load_catalog()
    assert len(catalog) > 300
    item = catalog.items[0]
    for field in ("id", "name", "url", "test_type", "keys"):
        assert field in item


def test_every_url_is_unique_and_resolvable():
    catalog = load_catalog()
    urls = [it["url"] for it in catalog.items]
    assert len(urls) == len(set(urls))
    assert all(catalog.has_url(u) for u in urls)


def test_validate_drops_hallucinated_urls():
    catalog = load_catalog()
    real = catalog.items[0]
    recs = [
        {"name": real["name"], "url": real["url"], "test_type": real["test_type"]},
        {"name": "Fake Test", "url": "https://www.shl.com/products/product-catalog/view/fake/", "test_type": "K"},
    ]
    clean = validate_recommendations(recs, catalog)
    assert len(clean) == 1
    assert clean[0]["url"] == real["url"]


def test_validate_clamps_to_ten():
    catalog = load_catalog()
    recs = [
        {"name": it["name"], "url": it["url"], "test_type": it["test_type"]}
        for it in catalog.items[:15]
    ]
    clean = validate_recommendations(recs, catalog)
    assert len(clean) == 10


def test_match_by_name_resolves_acronym():
    catalog = load_catalog()
    matches = catalog.match_by_name("OPQ32r", limit=1)
    assert matches and "opq" in matches[0]["name"].lower()
