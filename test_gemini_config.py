from service_testing.services.gemini_extractor import GeminiExtractorService
from service_testing.config import settings

def test_gemini_config():
    # Test that the configuration is picking up gemini-3.5-flash
    extractor = GeminiExtractorService()

    print(f"Gemini API Key configured: {bool(settings.gemini_api_key)}")
    print(f"Gemini Model from settings: {settings.gemini_model}")
    print(f"Extractor model name: {extractor.model_name}")

    # Verify it's using 3.5-flash
    assert extractor.model_name == "gemini-3.5-flash", f"Expected gemini-3.5-flash, got {extractor.model_name}"
    assert settings.gemini_model == "gemini-3.5-flash"

    # Check that client is initialized when API key is present
    if settings.gemini_api_key:
        assert extractor.client is not None, "Gemini client should be initialized when API key is present"
        print("✓ Gemini client initialized successfully")
    else:
        print("⚠ No Gemini API key found - client will be None (expected in test env)")

    print("Gemini configuration test passed!")

if __name__ == "__main__":
    test_gemini_config()