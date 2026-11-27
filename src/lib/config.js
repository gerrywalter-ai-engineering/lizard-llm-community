import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { ensureDirectory, expandHome } from "./paths.js";
import { getAppRoot, resolveAppPath, resolveAppRelativePath } from "./app-root.js";

const appRoot = getAppRoot();
const runtimeDirectory = process.env.LIZARD_RUNTIME_DIRECTORY
  ? expandHome(process.env.LIZARD_RUNTIME_DIRECTORY)
  : path.join(os.homedir(), ".lizard-llm-community");

// The one place the local app's port is decided. Everything else - the OAuth
// redirect URI, the launcher, the tray, the docs - derives from this, so the
// port can never drift between surfaces again.
//
// 7817 replaced 5567 in 1.2.48: 5567 is registered to another service and was
// colliding with unrelated software on real machines. 7817 is IANA-unassigned,
// is not the default for any common developer tool, and sits outside both the
// Windows excluded-port ranges and the ephemeral range.
//
// Overridable with LIZARD_PORT for anyone who still hits a conflict. NOTE: the
// OAuth redirect URI is registered with the identity provider, so a custom port
// only works if that URI is registered too - both 7817 and the legacy 5567 are.
export const DEFAULT_APP_PORT = 7817;
export const LEGACY_APP_PORT = 5567;

export function resolveAppPort(env = process.env) {
  const raw = env.LIZARD_PORT;
  if (raw == null || String(raw).trim() === "") return DEFAULT_APP_PORT;
  const parsed = Number.parseInt(String(raw), 10);
  return Number.isInteger(parsed) && parsed > 0 && parsed < 65536 ? parsed : DEFAULT_APP_PORT;
}

const appPort = resolveAppPort();

const defaultConfig = {
  appName: "lizard-llm",
  appPort,
  runtimeDirectory,
  defaultMode: "balanced",
  // Community Edition: no sign-in, no accounts, no licence server, no remote
  // controller. The keys stay so shared runtime code reads a defined shape.
  auth: {
    enabled: false,
    requiredForCommunity: false,
    issuer: null,
    clientId: null,
    apiAudience: null,
    apiBaseUrl: null,
    accountUrl: null,
    accountAuthBase: null,
    redirectUri: null,
    stateDirectory: path.join(runtimeDirectory, "auth"),
  },
  authorization: {
    remoteEntitlements: false,
  },
  remoteAgent: {
    enabled: false,
  },
  modelDirectory: path.join(runtimeDirectory, "models"),
  catalogPath: path.join(runtimeDirectory, "model-catalog.json"),
  maxRamUsagePercent: 85,
  maxVramUsagePercent: 90,
  benchmark: {
    prompt: "Summarize the trade-offs of local inference in four bullet points.",
    generatedTokens: 64,
    databasePath: path.join(runtimeDirectory, "benchmarks.sqlite"),
    ci: {
      model: null,
      prompt: "what is the difference between cats and dogs",
      maxTokens: 3000,
      warmupPrompt: "Reply with READY.",
      artifactDirectory: path.join(runtimeDirectory, "benchmark-results"),
      contextTokens: 4096,
      healthTimeoutSeconds: 180,
      requestTimeoutSeconds: 600,
      serverBasePort: 8090,
    },
  },
  generation: {
    maxTokens: 160,
    temperature: 0.2,
  },
  logging: {
    directory: path.join(runtimeDirectory, "logs"),
  },
  failureIntelligence: {
    enabled: true,
    directory: path.join(runtimeDirectory, "incidents"),
    repeatWindowMs: 10 * 60 * 1000,
    repeatThreshold: 3,
    requestTimeoutMs: 10_000,
    // Incidents are recorded locally only; nothing is ever sent.
  },
  product: {
    installationPath: path.join(runtimeDirectory, "installation.json"),
    enrollmentPath: path.join(runtimeDirectory, "enrollment.json"),
    fleetStatusPath: path.join(runtimeDirectory, "fleet-status.json"),
    onboardingPath: path.join(runtimeDirectory, "onboarding.json"),
    proDashboardStatePath: path.join(runtimeDirectory, "pro-dashboard-state.json"),
    profilesDirectory: path.join(runtimeDirectory, "profiles"),
    supportBundleDirectory: path.join(runtimeDirectory, "support-bundles"),
    supportCasesDirectory: path.join(runtimeDirectory, "support-cases"),
    benchmarkExportDirectory: path.join(runtimeDirectory, "benchmark-exports"),
    licensePath: path.join(runtimeDirectory, "license.json"),
    entitlementPath: path.join(runtimeDirectory, "entitlement.json"),
    updateHistoryPath: path.join(runtimeDirectory, "update-history.json"),
    updateDownloadsDirectory: path.join(runtimeDirectory, "update-downloads"),
    updateInstallDirectory: path.join(runtimeDirectory, "installed-updates"),
    auditDirectory: path.join(runtimeDirectory, "audit"),
    offlineDirectory: path.join(runtimeDirectory, "offline"),
    updates: {
      provider: "control-plane-manifest",
      owner: "gerrywalter-ai-engineering",
      repo: "lizard-llm-community",
      channel: "stable",
      enabled: false,
    },
    licensing: {
      publicKeyPath: path.join(runtimeDirectory, "commercial-public-key.pem"),
      offlineGraceDays: 30,
    },
    controlPlane: {
      enabled: false,
      tenantId: "local-default",
      directory: path.join(runtimeDirectory, "control-plane"),
      heartbeatHistoryLimit: 50,
      defaultSlaHours: {
        open: 24,
        investigating: 8,
        escalated: 4,
        resolved: 24,
      },
    },
  },
  engine: {
    installProvider: process.platform === "win32" ? "winget" : "manual",
    serverBinary: "llama-server",
    cliBinary: "llama-cli",
    completionBinary: "llama-completion",
    host: "127.0.0.1",
    port: 8080,
    native: {
      host: "127.0.0.1",
      port: 8013,
      runnerBinary: null,
      serverScript: path.join(appRoot, "src", "runtime", "server", "standalone.js"),
      buildDirectory: path.join(appRoot, "native", "lizard-native-runner", "build"),
    },
    caterpillar: {
      host: "127.0.0.1",
      port: 8014,
      runnerBinary: null,
      serverScript: path.join(appRoot, "src", "runtime", "server", "standalone.js"),
      buildDirectory: path.join(appRoot, "native", "caterpillar-runner", "build"),
    },
    preferredBinaryDirectories: process.platform === "win32"
      ? [path.join(os.homedir(), "AppData", "Local", "Microsoft", "WinGet", "Packages", "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe")]
      : [],
  },
  ollama: {
    modelsDirectory: path.join(os.homedir(), ".ollama", "models"),
    importStrategy: "hardlink",
    autoImport: true,
    autoPullIfMissing: false,
  },
};

function mergeObjects(base, override) {
  const output = { ...base };
  for (const [key, value] of Object.entries(override ?? {})) {
    if (value && typeof value === "object" && !Array.isArray(value) && base[key] && typeof base[key] === "object" && !Array.isArray(base[key])) {
      output[key] = mergeObjects(base[key], value);
      continue;
    }
    output[key] = value;
  }
  return output;
}

function readJsonIfPresent(filePath) {
  if (!filePath) {
    return {};
  }
  if (!fs.existsSync(filePath)) {
    return {};
  }
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

// Environment overrides for the auth block (non-secret values only). Env has
// the highest precedence so an operator can flip auth settings without
// editing config files; nothing here can be changed by a web page at runtime.
function applyAuthEnvOverrides(config) {
  const env = process.env;
  const auth = { ...config.auth };
  const authorization = { ...config.authorization };
  if (env.LIZARD_AUTH_ENABLED != null) {
    auth.enabled = ["1", "true", "yes"].includes(env.LIZARD_AUTH_ENABLED.toLowerCase());
  }
  if (env.LIZARD_AUTH_REQUIRED != null) {
    auth.requiredForCommunity = ["1", "true", "yes"].includes(env.LIZARD_AUTH_REQUIRED.toLowerCase());
  }
  if (env.LIZARD_AUTH_ISSUER) {
    auth.issuer = env.LIZARD_AUTH_ISSUER;
  }
  if (env.LIZARD_AUTH_CLIENT_ID) {
    auth.clientId = env.LIZARD_AUTH_CLIENT_ID;
  }
  if (env.LIZARD_AUTH_API_AUDIENCE) {
    auth.apiAudience = env.LIZARD_AUTH_API_AUDIENCE;
  }
  if (env.LIZARD_AUTH_API_BASE_URL) {
    auth.apiBaseUrl = env.LIZARD_AUTH_API_BASE_URL;
  }
  if (env.LIZARD_AUTH_ACCOUNT_URL) {
    auth.accountUrl = env.LIZARD_AUTH_ACCOUNT_URL;
  }
  if (env.LIZARD_AUTH_ACCOUNT_AUTH_BASE) {
    auth.accountAuthBase = env.LIZARD_AUTH_ACCOUNT_AUTH_BASE;
  }
  if (env.LIZARD_AUTH_REDIRECT_URI) {
    auth.redirectUri = env.LIZARD_AUTH_REDIRECT_URI;
  }
  if (env.LIZARD_AUTHZ_REMOTE_ENTITLEMENTS != null) {
    authorization.remoteEntitlements = ["1", "true", "yes"].includes(env.LIZARD_AUTHZ_REMOTE_ENTITLEMENTS.toLowerCase());
  }
  return { ...config, auth, authorization };
}

function applyFailureIntelligenceEnvOverrides(config) {
  const env = process.env;
  const current = config.failureIntelligence ?? {};
  const nodebb = { ...(current.nodebb ?? {}) };
  const email = { ...(current.email ?? {}) };
  let enabled = current.enabled !== false;
  if (env.LIZARD_FAILURE_INTELLIGENCE_ENABLED != null) {
    enabled = ["1", "true", "yes"].includes(env.LIZARD_FAILURE_INTELLIGENCE_ENABLED.toLowerCase());
  }
  if (env.LIZARD_NODEBB_URL) nodebb.baseUrl = env.LIZARD_NODEBB_URL;
  if (env.LIZARD_NODEBB_API_TOKEN) nodebb.apiToken = env.LIZARD_NODEBB_API_TOKEN;
  if (env.LIZARD_NODEBB_OPEN_QUESTIONS_CID) {
    nodebb.openQuestionsCategoryId = env.LIZARD_NODEBB_OPEN_QUESTIONS_CID;
  }
  if (env.LIZARD_INCIDENT_EMAIL_ENDPOINT) email.endpoint = env.LIZARD_INCIDENT_EMAIL_ENDPOINT;
  if (env.LIZARD_INCIDENT_EMAIL_API_TOKEN) email.apiToken = env.LIZARD_INCIDENT_EMAIL_API_TOKEN;
  if (env.LIZARD_INCIDENT_EMAIL_FROM) email.from = env.LIZARD_INCIDENT_EMAIL_FROM;
  if (env.LIZARD_INCIDENT_ADMIN_EMAILS) {
    email.adminTo = env.LIZARD_INCIDENT_ADMIN_EMAILS
      .split(",")
      .map((entry) => entry.trim())
      .filter(Boolean);
  }
  const userEmail = env.LIZARD_INCIDENT_USER_EMAIL || current.userEmail || null;
  return {
    ...config,
    failureIntelligence: {
      ...current,
      enabled,
      userEmail,
      nodebb,
      email,
    },
  };
}

function expandPathFields(config) {
  return {
    ...config,
    runtimeDirectory: expandHome(config.runtimeDirectory),
    modelDirectory: expandHome(config.modelDirectory),
    catalogPath: expandHome(config.catalogPath),
    auth: {
      ...config.auth,
      stateDirectory: expandHome(config.auth?.stateDirectory ?? path.join(runtimeDirectory, "auth")),
    },
    benchmark: {
      ...config.benchmark,
      databasePath: expandHome(config.benchmark.databasePath),
      ci: {
        ...(config.benchmark?.ci ?? {}),
        artifactDirectory: expandHome(config.benchmark?.ci?.artifactDirectory),
      },
    },
    logging: {
      ...config.logging,
      directory: expandHome(config.logging.directory),
    },
    failureIntelligence: {
      ...(config.failureIntelligence ?? {}),
      directory: expandHome(
        config.failureIntelligence?.directory ?? path.join(runtimeDirectory, "incidents"),
      ),
      nodebb: { ...(config.failureIntelligence?.nodebb ?? {}) },
      email: { ...(config.failureIntelligence?.email ?? {}) },
    },
    product: {
      ...config.product,
      installationPath: expandHome(config.product.installationPath),
      enrollmentPath: expandHome(config.product.enrollmentPath),
      fleetStatusPath: expandHome(config.product.fleetStatusPath),
      onboardingPath: expandHome(config.product.onboardingPath),
      proDashboardStatePath: expandHome(config.product.proDashboardStatePath),
      profilesDirectory: expandHome(config.product.profilesDirectory),
      supportBundleDirectory: expandHome(config.product.supportBundleDirectory),
      supportCasesDirectory: expandHome(config.product.supportCasesDirectory),
      benchmarkExportDirectory: expandHome(config.product.benchmarkExportDirectory),
      licensePath: expandHome(config.product.licensePath),
      entitlementPath: expandHome(config.product.entitlementPath),
      updateHistoryPath: expandHome(config.product.updateHistoryPath),
      updateDownloadsDirectory: expandHome(config.product.updateDownloadsDirectory),
      updateInstallDirectory: expandHome(config.product.updateInstallDirectory),
      auditDirectory: expandHome(config.product.auditDirectory),
      offlineDirectory: expandHome(config.product.offlineDirectory),
      licensing: {
        ...config.product.licensing,
        publicKeyPath: expandHome(config.product.licensing.publicKeyPath),
      },
      controlPlane: {
        ...config.product.controlPlane,
        directory: expandHome(config.product.controlPlane.directory),
      },
    },
    engine: {
      ...config.engine,
      native: {
        ...(config.engine.native ?? {}),
        runnerBinary: resolveAppRelativePath(expandHome(config.engine.native?.runnerBinary ?? null)),
        serverScript: resolveAppRelativePath(expandHome(config.engine.native?.serverScript ?? resolveAppPath("src", "runtime", "server", "standalone.js"))),
        buildDirectory: resolveAppRelativePath(expandHome(config.engine.native?.buildDirectory ?? resolveAppPath("native", "lizard-native-runner", "build"))),
      },
      caterpillar: {
        ...(config.engine.caterpillar ?? {}),
        runnerBinary: resolveAppRelativePath(expandHome(config.engine.caterpillar?.runnerBinary ?? null)),
        serverScript: resolveAppRelativePath(expandHome(config.engine.caterpillar?.serverScript ?? resolveAppPath("src", "runtime", "server", "standalone.js"))),
        buildDirectory: resolveAppRelativePath(expandHome(config.engine.caterpillar?.buildDirectory ?? resolveAppPath("native", "caterpillar-runner", "build"))),
      },
      preferredBinaryDirectories: (config.engine.preferredBinaryDirectories ?? []).map((entry) => expandHome(entry)),
    },
    ollama: {
      ...config.ollama,
      modelsDirectory: expandHome(config.ollama.modelsDirectory),
    },
  };
}

export function loadConfig() {
  const cwdConfig = path.resolve(process.cwd(), "lizard-llm.config.json");
  const appConfig = resolveAppPath("lizard-llm.config.json");
  const homeConfig = path.join(os.homedir(), ".lizard-llm", "config.json");
  const explicitConfig = process.env.LIZARD_CONFIG_PATH
    ? path.resolve(expandHome(process.env.LIZARD_CONFIG_PATH))
    : null;
  const config = expandPathFields(applyFailureIntelligenceEnvOverrides(applyAuthEnvOverrides(mergeObjects(
    defaultConfig,
    mergeObjects(
      readJsonIfPresent(homeConfig),
      mergeObjects(
        readJsonIfPresent(appConfig),
        mergeObjects(
          readJsonIfPresent(cwdConfig),
          readJsonIfPresent(explicitConfig),
        ),
      ),
    ),
  ))));
  ensureDirectory(config.runtimeDirectory);
  ensureDirectory(config.modelDirectory);
  ensureDirectory(config.benchmark.ci.artifactDirectory);
  ensureDirectory(config.logging.directory);
  ensureDirectory(config.failureIntelligence.directory);
  ensureDirectory(config.product.profilesDirectory);
  ensureDirectory(config.product.supportBundleDirectory);
  ensureDirectory(config.product.supportCasesDirectory);
  ensureDirectory(config.product.benchmarkExportDirectory);
  ensureDirectory(config.product.updateDownloadsDirectory);
  ensureDirectory(config.product.updateInstallDirectory);
  ensureDirectory(config.product.auditDirectory);
  ensureDirectory(config.product.offlineDirectory);
  ensureDirectory(config.product.controlPlane.directory);
  return config;
}
