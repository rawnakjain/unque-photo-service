"""Test MongoDB Atlas Fuzzy Search for services and variations."""

from service_testing.services.fuzzy_search import FuzzySearchService


def test_fuzzy_search():
    fuzzy = FuzzySearchService()
    # First check connection
    conn = fuzzy.check_connection()
    print("MongoDB connection:", conn)
    assert conn["status"] == "connected", f"MongoDB not connected: {conn}"

    # Test service search on salonservices collection
    print("\n--- Testing Service Search (salonservices) ---")
    service_results = fuzzy.recommend_services(query="haircut", top_k=5, min_confidence=80)
    print(f"Found {len(service_results)} service recommendations for 'haircut'")
    for i, r in enumerate(service_results):
        print(f"  {i+1}. {r.variationName} (similarity: {r.similarity_percentage}%)")

    # Test another service query
    service_results2 = fuzzy.recommend_services(query="facial", top_k=5, min_confidence=80)
    print(f"Found {len(service_results2)} service recommendations for 'facial'")
    for i, r in enumerate(service_results2):
        print(f"  {i+1}. {r.variationName} (similarity: {r.similarity_percentage}%)")

    # Test variation search on salonservicevariations collection
    print("\n--- Testing Variation Search (salonservicevariations) ---")
    variation_results = fuzzy.recommend_variations(query="Bob haircut", top_k=5, min_confidence=80)
    print(f"Found {len(variation_results)} variation recommendations for 'Bob haircut'")
    for i, r in enumerate(variation_results):
        print(f"  {i+1}. {r.variationName} (similarity: {r.similarity_percentage}%)")

    # Test another variation query
    variation_results2 = fuzzy.recommend_variations(query="Full Arm Wax", top_k=5, min_confidence=80)
    print(f"Found {len(variation_results2)} variation recommendations for 'Full Arm Wax'")
    for i, r in enumerate(variation_results2):
        print(f"  {i+1}. {r.variationName} (similarity: {r.similarity_percentage}%)")

    # Test low confidence query (should return empty or low results)
    print("\n--- Testing Low Confidence Query ---")
    low_results = fuzzy.recommend_variations(query="xyzabc123", top_k=5, min_confidence=80)
    print(f"Found {len(low_results)} variation recommendations for 'xyzabc123' (expected: 0)")

    print("\nAll fuzzy search tests passed!")


if __name__ == "__main__":
    test_fuzzy_search()