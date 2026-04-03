#!/usr/bin/env node
/**
 * Generate training data for the JS deobfuscation model.
 *
 * Sources:
 *   1. Ground-truth fixtures from ruvector-decompiler tests
 *   2. Synthetic minification of open-source npm packages
 *   3. Cross-version analysis patterns
 *
 * Output: JSONL where each line is:
 *   {"minified":"a$","original":"createRouter","context_strings":[...],"properties":[...],"kind":"function"}
 *
 * Usage:
 *   node scripts/training/generate-deobfuscation-data.mjs [--output training-data.jsonl] [--min-pairs 10000]
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from "fs";
import { join, resolve, extname } from "path";
import { execSync } from "child_process";
import { parseArgs } from "util";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const { values: args } = parseArgs({
  options: {
    output: { type: "string", default: "training-data.jsonl" },
    "min-pairs": { type: "string", default: "10000" },
    "skip-npm": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
});

if (args.help) {
  console.log("Usage: generate-deobfuscation-data.mjs [--output FILE] [--min-pairs N] [--skip-npm]");
  process.exit(0);
}

const OUTPUT_PATH = resolve(args.output);
const MIN_PAIRS = parseInt(args["min-pairs"], 10);

/** @type {Array<{minified: string, original: string, context_strings: string[], properties: string[], kind: string}>} */
const pairs = [];

// ---------------------------------------------------------------------------
// Source 1: Ground-truth fixtures
// ---------------------------------------------------------------------------

function extractGroundTruthFixtures() {
  const ROOT = resolve(import.meta.dirname, "../../crates/ruvector-decompiler/tests");
  const files = ["ground_truth.rs", "real_world.rs"];

  for (const file of files) {
    const path = join(ROOT, file);
    let content;
    try {
      content = readFileSync(path, "utf8");
    } catch {
      console.warn(`  [skip] ${path} not found`);
      continue;
    }

    // Extract (&str, &str) pairs from ORIGINAL_NAMES arrays.
    // Pattern: ("minified", "original")
    const tupleRe = /\("([^"]+)",\s*"([^"]+)"\)/g;
    let match;
    while ((match = tupleRe.exec(content)) !== null) {
      const [, minified, original] = match;
      if (minified.length <= 3 && original.length > 3) {
        pairs.push({
          minified,
          original,
          context_strings: [],
          properties: [],
          kind: "var",
        });
      }
    }

    // Extract standalone name arrays: &["Router", "Request", ...]
    const nameArrayRe = /ORIGINAL_NAMES:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/g;
    while ((match = nameArrayRe.exec(content)) !== null) {
      const names = match[1].match(/"([^"]+)"/g);
      if (names) {
        names.forEach((n, i) => {
          const original = n.replace(/"/g, "");
          const minified = String.fromCharCode(97 + (i % 26));
          if (!pairs.some((p) => p.original === original && p.minified === minified)) {
            pairs.push({
              minified,
              original,
              context_strings: [],
              properties: [],
              kind: "function",
            });
          }
        });
      }
    }

    // Extract string literals from minified source constants for context.
    const strLitRe = /"([a-zA-Z_][a-zA-Z0-9_]{2,})"/g;
    const contextStrings = new Set();
    while ((match = strLitRe.exec(content)) !== null) {
      const s = match[1];
      if (!["var", "let", "const", "function", "class", "return"].includes(s)) {
        contextStrings.add(s);
      }
    }

    // Enrich pairs from this file with context strings.
    const ctxArray = [...contextStrings].slice(0, 20);
    for (const pair of pairs) {
      if (pair.context_strings.length === 0) {
        pair.context_strings = ctxArray.slice(0, 5);
      }
    }
  }

  console.log(`  [ground-truth] extracted ${pairs.length} pairs`);
}

// ---------------------------------------------------------------------------
// Source 2: Synthetic minification from common identifier patterns
// ---------------------------------------------------------------------------

/**
 * Generate synthetic training pairs from common JS identifier patterns.
 * This simulates what real minifiers produce.
 */
function generateSyntheticPairs() {
  const COMMON_NAMES = {
    function: [
      "createElement", "appendChild", "removeChild", "setAttribute",
      "addEventListener", "removeEventListener", "querySelector", "querySelectorAll",
      "getElementById", "getElementsByClassName", "preventDefault", "stopPropagation",
      "dispatch", "subscribe", "unsubscribe", "connect", "disconnect",
      "initialize", "configure", "validate", "serialize", "deserialize",
      "transform", "normalize", "sanitize", "encode", "decode",
      "encrypt", "decrypt", "compress", "decompress",
      "fetchData", "postData", "getData", "setData", "deleteData",
      "handleClick", "handleSubmit", "handleChange", "handleError",
      "createRouter", "createStore", "createContext", "createRef",
      "useEffect", "useState", "useCallback", "useMemo", "useReducer",
      "parseJSON", "stringifyJSON", "parseURL", "formatDate",
      "sortArray", "filterItems", "mapValues", "reduceTotal",
      "debounce", "throttle", "memoize", "curry",
      "deepClone", "deepMerge", "deepEqual", "shallowEqual",
      "getToken", "setToken", "clearToken", "refreshToken",
      "openModal", "closeModal", "toggleMenu", "scrollToTop",
      "sendRequest", "cancelRequest", "retryRequest",
      "renderComponent", "mountComponent", "unmountComponent",
      "logMessage", "logError", "logWarning", "logInfo",
      "readFile", "writeFile", "deleteFile", "listFiles",
      "startServer", "stopServer", "restartServer",
      "connectDatabase", "queryDatabase", "closeConnection",
      "hashPassword", "verifyPassword", "generateSalt",
      "createSession", "destroySession", "getSession",
      "emitEvent", "onEvent", "offEvent", "broadcastEvent",
      "parseTemplate", "renderTemplate", "compileTemplate",
      "formatCurrency", "formatNumber", "formatPercentage",
      "calculateTotal", "calculateTax", "calculateDiscount",
      "validateEmail", "validatePhone", "validatePassword",
      "uploadFile", "downloadFile", "processFile",
    ],
    class: [
      "Component", "Controller", "Service", "Factory", "Repository",
      "Manager", "Handler", "Builder", "Parser", "Formatter",
      "Validator", "Serializer", "Transformer", "Adapter", "Wrapper",
      "EventEmitter", "Observable", "Iterator", "Generator",
      "HttpClient", "WebSocketClient", "DatabaseClient",
      "UserService", "AuthService", "DataService", "CacheService",
      "Router", "Middleware", "Pipeline", "Queue", "Stack",
      "Logger", "Monitor", "Tracker", "Analyzer",
      "Config", "Settings", "Options", "Preferences",
      "Request", "Response", "Context", "Session",
      "Model", "View", "Presenter", "ViewModel",
      "Store", "State", "Reducer", "Action",
      "Plugin", "Extension", "Module", "Package",
    ],
    var: [
      "config", "options", "settings", "preferences", "defaults",
      "state", "props", "context", "params", "args",
      "result", "output", "response", "data", "payload",
      "error", "message", "status", "code", "type",
      "name", "label", "title", "description", "content",
      "items", "list", "array", "collection", "set",
      "map", "table", "index", "cache", "buffer",
      "count", "total", "sum", "average", "max", "min",
      "width", "height", "size", "length", "offset",
      "timeout", "interval", "delay", "duration",
      "callback", "handler", "listener", "observer",
      "template", "pattern", "schema", "format",
      "prefix", "suffix", "separator", "delimiter",
      "source", "target", "origin", "destination",
      "parent", "child", "root", "node", "element",
      "key", "value", "pair", "entry", "record",
      "token", "secret", "hash", "salt", "nonce",
      "baseUrl", "endpoint", "apiKey", "apiVersion",
      "currentUser", "currentPage", "currentIndex",
      "isLoading", "isValid", "isActive", "isVisible",
      "hasError", "hasChanges", "hasPermission",
    ],
  };

  // Context strings commonly found near specific identifier types.
  const CONTEXT_MAP = {
    createElement: ["div", "span", "button", "input", "innerHTML"],
    addEventListener: ["click", "submit", "change", "keydown", "DOMContentLoaded"],
    fetchData: ["GET", "POST", "Content-Type", "application/json", "Authorization"],
    createRouter: ["GET", "POST", "route", "middleware", "path"],
    useState: ["setState", "initialState", "render", "component"],
    parseJSON: ["JSON", "parse", "stringify", "object", "string"],
    connectDatabase: ["connection", "host", "port", "database", "query"],
    hashPassword: ["bcrypt", "salt", "rounds", "hash", "verify"],
    validateEmail: ["email", "regex", "pattern", "valid", "invalid"],
    HttpClient: ["fetch", "XMLHttpRequest", "headers", "method", "body"],
    Router: ["route", "path", "handler", "middleware", "GET"],
    EventEmitter: ["emit", "on", "off", "once", "listeners"],
    Logger: ["log", "error", "warn", "info", "debug"],
    Store: ["state", "dispatch", "subscribe", "getState", "reducer"],
  };

  // Property access patterns.
  const PROPERTY_MAP = {
    createElement: ["tagName", "className", "id", "style"],
    fetchData: ["method", "headers", "body", "status"],
    createRouter: ["method", "path", "handler", "params"],
    Router: ["routes", "middleware", "use", "get", "post"],
    Component: ["props", "state", "render", "componentDidMount"],
    Store: ["state", "dispatch", "subscribe", "getState"],
    Logger: ["level", "message", "timestamp", "format"],
    config: ["host", "port", "database", "username"],
  };

  // Minifier name generators.
  const minifierStyles = [
    (i) => String.fromCharCode(97 + (i % 26)),                         // a, b, c...
    (i) => String.fromCharCode(97 + (i % 26)) + "$",                   // a$, b$...
    (i) => "_" + String.fromCharCode(97 + (i % 26)),                   // _a, _b...
    (i) => "_0x" + (0x1a2b + i).toString(16),                          // _0x1a2b...
    (i) => String.fromCharCode(97 + (i % 26)) + (i % 10).toString(),   // a0, b1...
    (i) => "__" + String.fromCharCode(97 + (i % 26)),                  // __a, __b...
    (i) => "$" + String.fromCharCode(97 + (i % 26)),                   // $a, $b...
    (i) => String.fromCharCode(65 + (i % 26)),                         // A, B, C...
  ];

  let syntheticCount = 0;

  for (const [kind, names] of Object.entries(COMMON_NAMES)) {
    for (let i = 0; i < names.length; i++) {
      const original = names[i];
      // Generate multiple minified variants per original name.
      const numVariants = Math.min(3, minifierStyles.length);
      for (let v = 0; v < numVariants; v++) {
        const styleIdx = (i + v) % minifierStyles.length;
        const minified = minifierStyles[styleIdx](i);
        const ctx = CONTEXT_MAP[original] || [];
        const props = PROPERTY_MAP[original] || [];

        pairs.push({
          minified,
          original,
          context_strings: ctx.length > 0 ? ctx : generateGenericContext(original),
          properties: props.length > 0 ? props : generateGenericProperties(kind),
          kind,
        });
        syntheticCount++;
      }
    }
  }

  console.log(`  [synthetic] generated ${syntheticCount} pairs`);
}

/**
 * Generate generic context strings from an identifier name.
 * Splits camelCase into tokens and uses them as context hints.
 */
function generateGenericContext(name) {
  const tokens = name
    .replace(/([A-Z])/g, " $1")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2);
  return tokens.slice(0, 5);
}

/**
 * Generate generic property names based on declaration kind.
 */
function generateGenericProperties(kind) {
  switch (kind) {
    case "function":
      return ["length", "name", "call", "apply"];
    case "class":
      return ["prototype", "constructor", "name"];
    case "var":
      return ["toString", "valueOf"];
    default:
      return [];
  }
}

// ---------------------------------------------------------------------------
// Source 3: Cross-version augmentation
// ---------------------------------------------------------------------------

/**
 * Generate augmented pairs by simulating cross-version name changes.
 * Same original name gets different minified names across "versions".
 */
function generateCrossVersionPairs() {
  const existingOriginals = [...new Set(pairs.map((p) => p.original))];
  let augmented = 0;

  for (const original of existingOriginals) {
    const existing = pairs.find((p) => p.original === original);
    if (!existing) continue;

    // Simulate 2-3 additional "versions" with different minified names.
    const versions = 2 + Math.floor(Math.random() * 2);
    for (let v = 0; v < versions; v++) {
      const minified = generateRandomMinifiedName();
      if (pairs.some((p) => p.minified === minified && p.original === original)) continue;

      pairs.push({
        minified,
        original,
        context_strings: existing.context_strings,
        properties: existing.properties,
        kind: existing.kind,
      });
      augmented++;
    }
  }

  console.log(`  [cross-version] augmented ${augmented} pairs`);
}

/**
 * Generate a random minified-style variable name.
 */
function generateRandomMinifiedName() {
  const styles = [
    () => {
      const c = String.fromCharCode(97 + Math.floor(Math.random() * 26));
      return c + Math.floor(Math.random() * 100);
    },
    () => "_0x" + Math.floor(Math.random() * 0xffff).toString(16),
    () => {
      const a = String.fromCharCode(97 + Math.floor(Math.random() * 26));
      const b = String.fromCharCode(97 + Math.floor(Math.random() * 26));
      return a + b;
    },
    () => "$" + String.fromCharCode(97 + Math.floor(Math.random() * 26)),
  ];
  return styles[Math.floor(Math.random() * styles.length)]();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("Generating deobfuscation training data...\n");

console.log("Source 1: Ground-truth fixtures");
extractGroundTruthFixtures();

console.log("\nSource 2: Synthetic minification patterns");
generateSyntheticPairs();

console.log("\nSource 3: Cross-version augmentation");
generateCrossVersionPairs();

// Deduplicate.
const seen = new Set();
const deduplicated = pairs.filter((p) => {
  const key = `${p.minified}|${p.original}`;
  if (seen.has(key)) return false;
  seen.add(key);
  return true;
});

console.log(`\nTotal: ${deduplicated.length} unique pairs (target: ${MIN_PAIRS})`);

if (deduplicated.length < MIN_PAIRS) {
  console.warn(`WARNING: Only ${deduplicated.length} pairs generated, below target of ${MIN_PAIRS}.`);
  console.warn("Consider adding more npm packages or expanding COMMON_NAMES.");
}

// Shuffle for training.
for (let i = deduplicated.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [deduplicated[i], deduplicated[j]] = [deduplicated[j], deduplicated[i]];
}

// Write JSONL.
const lines = deduplicated.map((p) => JSON.stringify(p)).join("\n");
writeFileSync(OUTPUT_PATH, lines + "\n", "utf8");
console.log(`\nWrote ${deduplicated.length} training pairs to ${OUTPUT_PATH}`);
