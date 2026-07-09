"""Visits the deployed Streamlit app so it doesn't go to sleep from inactivity.
"""

import sys

from playwright.sync_api import sync_playwright

APP_URL = "https://ml-trading.streamlit.app/"


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        try:
            page.goto(APP_URL, wait_until="networkidle", timeout=60_000)
        except Exception as e:
            print(f"Failed to load {APP_URL}: {e}")
            browser.close()
            return 1

        page.wait_for_timeout(5_000)

        wake_button = page.get_by_text("get this app back up", exact=False)
        if wake_button.count() > 0:
            wake_button.first.click()
            page.wait_for_timeout(15_000)
            print("App was asleep — clicked the wake-up button.")
        else:
            print("App was already awake.")

        browser.close()
        return 0


if __name__ == "__main__":
    sys.exit(main())
