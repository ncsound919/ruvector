/**
 * @ruvector/ruvbot - Self-learning AI assistant bot
 *
 * High-performance bot with WASM embeddings, vector memory,
 * and multi-platform integrations (Slack, Discord, webhooks).
 *
 * @example Quick Start
 * ```typescript
 * import { RuvBot } from '@ruvector/ruvbot';
 *
 * const bot = new RuvBot({
 *   name: 'MyBot',
 *   llm: { provider: 'anthropic' },
 *   slack: { enabled: true }
 * });
 *
 * await bot.start();
 * const response = await bot.chat('Hello!');
 * console.log(response.content);
 * ```
 *
 * @example With Memory
 * ```typescript
 * import { RuvBot } from '@ruvector/ruvbot';
 *
 * const bot = new RuvBot({
 *   memory: {
 *     dimensions: 384,
 *     maxVectors: 100000,
 *     indexType: 'hnsw'
 *   }
 * });
 *
 * await bot.start();
 *
 * // Store knowledge
 * await bot.storeInMemory('Important fact about X', {
 *   source: 'external',
 *   tags: ['facts', 'important']
 * });
 *
 * // Search memory
 * const memories = await bot.searchMemory('tell me about X');
 * ```
 *
 * @example With Skills
 * ```typescript
 * import { RuvBot, SkillEntity } from '@ruvector/ruvbot';
 *
 * const bot = new RuvBot();
 *
 * // Register custom skill
 * const weatherSkill = SkillEntity.create(
 *   'weather',
 *   'Get current weather for a location',
 *   async (input, context) => {
 *     const { location } = input;
 *     // Fetch weather...
 *     return { success: true, data: { temp: 72, condition: 'sunny' } };
 *   }
 * );
 *
 * await bot.start();
 * ```
 *
 * @packageDocumentation
 */

// Main RuvBot class
export { RuvBot, RuvBotOptions, ChatOptions, ChatResponse } from './RuvBot.js';

// Core types and entities
export * from './core/types.js';
export * from './core/errors.js';
export * from './core/entities/index.js';
export { BotConfig, BotConfigSchema, ConfigManager } from './core/BotConfig.js';
export { BotStateManager, BotStatus, BotMetrics } from './core/BotState.js';

// Utilities
export { createLogger, Logger, LogLevel } from './utils/logger.js';

// Default export
export { RuvBot as default } from './RuvBot.js';
