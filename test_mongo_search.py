from service_testing.services.mongo_search import MongoSearchService

def test_hybrid_search():
    mongo = MongoSearchService()
    # First check connection
    conn = mongo.check_connection()
    print("MongoDB connection:", conn)
    assert conn["status"] == "connected", f"MongoDB not connected: {conn}"

    # Test a known query from the notebook: "Bob haircut"
    results = mongo.hybrid_search(query="Bob haircut", top_k=5)
    print(f"Found {len(results)} results for 'Bob haircut'")
    for i, r in enumerate(results):
        print(f"  {i+1}. {r.get('variationName')} (score: {r.get('score')})")

    assert len(results) > 0, "Should return at least one result for 'Bob haircut'"

    # Test another query: "Full Arm"
    results2 = mongo.hybrid_search(query="Full Arm", top_k=5)
    print(f"Found {len(results2)} results for 'Full Arm'")
    assert len(results2) > 0, "Should return at least one result for 'Full Arm'"

    # Test the recommend_variations method which returns typed models
    recs = mongo.recommend_variations(query="Bridal Makeup", top_k=5)
    print(f"Recommendations for 'Bridal Makeup': {len(recs)}")
    for rec in recs:
        print(f"  - {rec.variationName} ({rec.similarity_percentage}%)")

    assert len(recs) > 0, "Should return recommendations for 'Bridal Makeup'"

if __name__ == "__main__":
    test_hybrid_search()
    print("All MongoDB search tests passed!")