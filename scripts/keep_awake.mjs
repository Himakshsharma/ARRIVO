// Opens the live Streamlit app in a headless browser so it never hibernates.
// If the app is already asleep, clicks the "Yes, get this app back up!" button.
import { chromium } from "playwright";

const URL = process.env.APP_URL || "https://arrivo-eta.streamlit.app/";
const browser = await chromium.launch();
const page = await browser.newPage();
try {
  await page.goto(URL, { waitUntil: "domcontentloaded", timeout: 120000 });
  await page.waitForTimeout(8000);
  const wake = page.getByRole("button", { name: /get this app back up/i });
  if (await wake.count()) {
    console.log("App was asleep, waking it up...");
    await wake.first().click();
    await page.waitForTimeout(90000);   // give it time to boot
  }
  // The app itself runs inside an iframe; wait for its header to confirm it's alive.
  const app = page.frameLocator("iframe[title='streamlitApp']");
  await app.getByText("Know which trucks will be late").first().waitFor({ timeout: 180000 });
  console.log("Arrivo is awake:", URL);
} finally {
  await browser.close();
}
