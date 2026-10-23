"""
test_app_pages.py -- Comprehensive automated test using Streamlit AppTest
to simulate navigating through all 10 pages and ensuring zero exceptions.
"""
from streamlit.testing.v1 import AppTest
import os

def test_all_pages():
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    at = AppTest.from_file(app_path).run()
    assert not at.exception, f"Overview page encountered exception: {at.exception}"
    print("[PASS] Page 1: Overview rendered cleanly.")

    # Radio button navigation
    pages = [
        "2. How It Works (Decay Math)",
        "3. Financial Equities (SPY)",
        "4. Macro Consumer (PCE)",
        "5. Epidemiology (COVID-19)",
        "6. Cryptocurrency (BTC / ETH)",
        "7. Walk-Forward Validation",
        "8. Benchmarks & Trading",
        "9. Methodology & Integrity",
        "10. Conclusions & Roadmap",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, f"Page '{p}' encountered exception: {at.exception}"
        print(f"[PASS] Page '{p}' rendered cleanly.")

    print("\nALL 10 PAGES TESTED AND VERIFIED WITH ZERO RUNTIME EXCEPTIONS!")

if __name__ == "__main__":
    test_all_pages()
