/**
 * Bot configuration management
 */

import { z } from 'zod';
import type { LLMProvider, MemoryConfig, Platform } from './types.js';

// ============================================================================
// Configuration Schema
// ============================================================================

export const MemoryConfigSchema = z.object({
  dimensions: z.number().int().min(64).max(4096).default(384),
  maxVectors: z.number().int().min(1000).max(10000000).default(100000),
  indexType: z.enum(['hnsw', 'flat', 'ivf']).default('hnsw'),
  persistPath: z.string().optional(),
  efConstruction: z.number().int().min(16).max(500).default(200),
  efSearch: z.number().int().min(10).max(500).default(50),
  m: z.number().int().min(4).max(64).default(16),
});

export const LLMConfigSchema = z.object({
  provider: z.enum(['anthropic', 'openai', 'google', 'local', 'ruvllm']).default('anthropic'),
  model: z.string().default('claude-sonnet-4-20250514'),
  apiKey: z.string().optional(),
  baseUrl: z.string().url().optional(),
  temperature: z.number().min(0).max(2).default(0.7),
  maxTokens: z.number().int().min(1).max(200000).default(4096),
  streaming: z.boolean().default(true),
});

export const SlackConfigSchema = z.object({
  enabled: z.boolean().default(false),
  botToken: z.string().optional(),
  signingSecret: z.string().optional(),
  appToken: z.string().optional(),
  socketMode: z.boolean().default(true),
});

export const DiscordConfigSchema = z.object({
  enabled: z.boolean().default(false),
  token: z.string().optional(),
  clientId: z.string().optional(),
  guildId: z.string().optional(),
});

export const WebhookConfigSchema = z.object({
  enabled: z.boolean().default(false),
  secret: z.string().optional(),
  endpoints: z.array(z.string().url()).default([]),
});

export const APIConfigSchema = z.object({
  enabled: z.boolean().default(true),
  port: z.number().int().min(1).max(65535).default(3000),
  host: z.string().default('0.0.0.0'),
  cors: z.boolean().default(true),
  rateLimit: z.object({
    max: z.number().int().default(100),
    timeWindow: z.number().int().default(60000),
  }).default({}),
  auth: z.object({
    enabled: z.boolean().default(false),
    type: z.enum(['bearer', 'basic', 'apikey']).default('bearer'),
    secret: z.string().optional(),
  }).default({}),
});

export const StorageConfigSchema = z.object({
  type: z.enum(['sqlite', 'postgres', 'memory']).default('sqlite'),
  path: z.string().default('./data/ruvbot.db'),
  connectionString: z.string().optional(),
  poolSize: z.number().int().min(1).max(100).default(10),
});

export const LoggingConfigSchema = z.object({
  level: z.enum(['trace', 'debug', 'info', 'warn', 'error', 'fatal']).default('info'),
  pretty: z.boolean().default(true),
  file: z.string().optional(),
});

export const BotConfigSchema = z.object({
  name: z.string().min(1).max(64).default('RuvBot'),
  version: z.string().default('0.1.0'),
  environment: z.enum(['development', 'staging', 'production']).default('development'),

  // Core settings
  memory: MemoryConfigSchema.default({}),
  llm: LLMConfigSchema.default({}),
  storage: StorageConfigSchema.default({}),
  logging: LoggingConfigSchema.default({}),
  api: APIConfigSchema.default({}),

  // Integrations
  slack: SlackConfigSchema.default({}),
  discord: DiscordConfigSchema.default({}),
  webhook: WebhookConfigSchema.default({}),

  // Skills
  skills: z.object({
    enabled: z.array(z.string()).default(['search', 'summarize', 'code', 'memory']),
    custom: z.array(z.string()).default([]),
    directory: z.string().default('./skills'),
  }).default({}),

  // Session settings
  session: z.object({
    defaultTTL: z.number().int().min(60000).default(3600000), // 1 hour
    maxPerUser: z.number().int().min(1).max(100).default(10),
    maxMessages: z.number().int().min(10).max(10000).default(1000),
  }).default({}),

  // Worker settings
  workers: z.object({
    poolSize: z.number().int().min(1).max(50).default(4),
    taskTimeout: z.number().int().min(1000).default(30000),
    retryAttempts: z.number().int().min(0).max(10).default(3),
  }).default({}),
});

export type BotConfig = z.infer<typeof BotConfigSchema>;

// ============================================================================
// Configuration Manager
// ============================================================================

export class ConfigManager {
  private config: BotConfig;

  constructor(initialConfig?: Partial<BotConfig>) {
    this.config = BotConfigSchema.parse(initialConfig ?? {});
  }

  /**
   * Get the full configuration
   */
  getConfig(): Readonly<BotConfig> {
    return Object.freeze({ ...this.config });
  }

  /**
   * Get a specific configuration section
   */
  get<K extends keyof BotConfig>(key: K): BotConfig[K] {
    return this.config[key];
  }

  /**
   * Update configuration
   */
  update(updates: Partial<BotConfig>): void {
    this.config = BotConfigSchema.parse({
      ...this.config,
      ...updates,
    });
  }

  /**
   * Validate configuration
   */
  validate(): { valid: boolean; errors: string[] } {
    try {
      BotConfigSchema.parse(this.config);
      return { valid: true, errors: [] };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          valid: false,
          errors: error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
        };
      }
      throw error;
    }
  }

  /**
   * Load configuration from environment variables
   */
  static fromEnv(): ConfigManager {
    const config: Partial<BotConfig> = {};

    // LLM configuration
    if (process.env.ANTHROPIC_API_KEY) {
      config.llm = {
        ...config.llm,
        provider: 'anthropic',
        apiKey: process.env.ANTHROPIC_API_KEY,
      };
    } else if (process.env.OPENAI_API_KEY) {
      config.llm = {
        ...config.llm,
        provider: 'openai',
        apiKey: process.env.OPENAI_API_KEY,
      };
    }

    // Slack configuration
    if (process.env.SLACK_BOT_TOKEN) {
      config.slack = {
        enabled: true,
        botToken: process.env.SLACK_BOT_TOKEN,
        signingSecret: process.env.SLACK_SIGNING_SECRET,
        appToken: process.env.SLACK_APP_TOKEN,
        socketMode: true,
      };
    }

    // Discord configuration
    if (process.env.DISCORD_TOKEN) {
      config.discord = {
        enabled: true,
        token: process.env.DISCORD_TOKEN,
        clientId: process.env.DISCORD_CLIENT_ID,
        guildId: process.env.DISCORD_GUILD_ID,
      };
    }

    // API configuration
    if (process.env.RUVBOT_PORT) {
      config.api = {
        ...config.api,
        port: parseInt(process.env.RUVBOT_PORT, 10),
      };
    }

    // Storage configuration
    if (process.env.DATABASE_URL) {
      config.storage = {
        type: 'postgres',
        connectionString: process.env.DATABASE_URL,
        path: '',
      };
    }

    // Logging
    if (process.env.RUVBOT_LOG_LEVEL) {
      config.logging = {
        ...config.logging,
        level: process.env.RUVBOT_LOG_LEVEL as 'debug' | 'info' | 'warn' | 'error',
      };
    }

    return new ConfigManager(config);
  }

  /**
   * Export configuration as JSON
   */
  toJSON(): string {
    return JSON.stringify(this.config, null, 2);
  }
}
