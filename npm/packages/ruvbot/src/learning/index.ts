/**
 * Learning module exports
 *
 * Provides WASM embeddings, vector memory, and LLM orchestration.
 */

// Placeholder exports - to be implemented
export const LEARNING_MODULE_VERSION = '0.1.0';

// Memory management (to be implemented)
export interface MemoryManagerOptions {
  dimensions: number;
  maxVectors: number;
  indexType: 'hnsw' | 'flat' | 'ivf';
  persistPath?: string;
}

// Embeddings (to be implemented)
export interface EmbedderOptions {
  model?: string;
  dimensions?: number;
  batchSize?: number;
}

// LLM orchestration (to be implemented)
export interface LLMOptions {
  provider: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
}
