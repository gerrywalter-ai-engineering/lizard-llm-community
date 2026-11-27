// Lizard LLM Community Edition.
//
// This open-source build has no accounts, credits, licence checks or
// telemetry: every feature that ships here is free and works offline. The
// paid Lizard LLM app (https://lizard-llm.qendryx.com) adds the Benchmark
// suite, the Benchmark Dashboard and Remote access on top of the same engines.

export const EDITION = "community";

export const PRO_URL = "https://lizard-llm.qendryx.com";
export const PRO_DOWNLOAD_URL = "https://lizard-llm.qendryx.com/download.html";

// What the one-time EUR 5 upgrade adds. Shown in the onboarding Pro card and
// in `lizard pro`. Keep in sync with the paid product page.
export const PRO_OFFER = Object.freeze({
  price: "€5",
  billing: "one-time payment, no subscription",
  url: PRO_URL,
  downloadUrl: PRO_DOWNLOAD_URL,
  includes: Object.freeze([
    "Signed one-click Windows installer with automatic updates",
    "Benchmark Mode: Lizard vs Ollama vs llama.cpp, Q1–Q9 side by side",
    "Benchmark Dashboard with charts, leaderboard and exportable reports",
    "Remote: pair another PC and run models on it from this machine",
    "Five days of Benchmark access included with the first payment",
    "Supports the development of both open-source engines",
  ]),
});

// Community Edition never gates a feature. The function keeps the call sites
// of the shared runtime unchanged and always reports an open plan.
export function getCreditState() {
  return {
    edition: EDITION,
    mode: "community",
    allowed: true,
    pro: false,
    proOffer: PRO_OFFER,
  };
}

export function assertProPlanFeatureAllowed() {
  return getCreditState();
}
