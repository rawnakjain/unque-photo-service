from service_testing.services.gemini_extractor import GeminiExtractorService
from service_testing.config import settings

def test_gemini_init():
    # Test that the Gemini extractor initializes correctly
    extractor = GeminiExtractorService()

    print(f"Gemini API Key configured: {bool(settings.gemini_api_key)}")
    print(f"Gemini Model from settings: {settings.gemini_model}")
    print(f"Extractor model name: {extractor.model_name}")
    print(f"Extractor client initialized: {extractor.client is not None}")

    # Verify it's using 3.5-flash
    assert extractor.model_name == "gemini-3.5-flash", f"Expected gemini-3.5-flash, got {extractor.model_name}"
    assert settings.gemini_model == "gemini-3.5-flash"

    print("✓ Gemini extractor initialization test passed!")

if __name__ == "__main__":
    test_gemini_init()