/**
 * Core types and interfaces for agentic-synth
 */
import { z } from 'zod';
export type JsonPrimitive = string | number | boolean | null;
export type JsonArray = JsonValue[];
export type JsonObject = {
    [key: string]: JsonValue;
};
export type JsonValue = JsonPrimitive | JsonArray | JsonObject;
export interface SchemaField {
    type: string;
    required?: boolean;
    properties?: Record<string, SchemaField>;
    items?: SchemaField;
    enum?: unknown[];
    minimum?: number;
    maximum?: number;
    pattern?: string;
}
export type DataSchema = Record<string, SchemaField>;
export type DataConstraints = Record<string, unknown>;
export declare const ModelProviderSchema: z.ZodEnum<{
    gemini: "gemini";
    openrouter: "openrouter";
}>;
export type ModelProvider = z.infer<typeof ModelProviderSchema>;
export declare const CacheStrategySchema: z.ZodEnum<{
    none: "none";
    memory: "memory";
    disk: "disk";
}>;
export type CacheStrategy = z.infer<typeof CacheStrategySchema>;
export declare const DataTypeSchema: z.ZodEnum<{
    json: "json";
    text: "text";
    timeseries: "timeseries";
    events: "events";
    structured: "structured";
    csv: "csv";
}>;
export type DataType = z.infer<typeof DataTypeSchema>;
export interface SynthConfig {
    provider: ModelProvider;
    apiKey?: string;
    model?: string;
    cacheStrategy?: CacheStrategy;
    cacheTTL?: number;
    maxRetries?: number;
    timeout?: number;
    streaming?: boolean;
    automation?: boolean;
    vectorDB?: boolean;
    enableFallback?: boolean;
    fallbackChain?: ModelProvider[];
}
export declare const SynthConfigSchema: z.ZodObject<{
    provider: z.ZodEnum<{
        gemini: "gemini";
        openrouter: "openrouter";
    }>;
    apiKey: z.ZodOptional<z.ZodString>;
    model: z.ZodOptional<z.ZodString>;
    cacheStrategy: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        none: "none";
        memory: "memory";
        disk: "disk";
    }>>>;
    cacheTTL: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    maxRetries: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    timeout: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    streaming: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    automation: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    vectorDB: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    enableFallback: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    fallbackChain: z.ZodOptional<z.ZodArray<z.ZodEnum<{
        gemini: "gemini";
        openrouter: "openrouter";
    }>>>;
}, z.core.$strip>;
export interface GeneratorOptions {
    count?: number;
    schema?: DataSchema;
    format?: 'json' | 'csv' | 'array';
    seed?: string | number;
    constraints?: DataConstraints;
}
export declare const GeneratorOptionsSchema: z.ZodObject<{
    count: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    schema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        json: "json";
        csv: "csv";
        array: "array";
    }>>>;
    seed: z.ZodOptional<z.ZodUnion<readonly [z.ZodString, z.ZodNumber]>>;
    constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, z.core.$strip>;
export interface TimeSeriesOptions extends GeneratorOptions {
    startDate?: Date | string;
    endDate?: Date | string;
    interval?: string;
    metrics?: string[];
    trend?: 'up' | 'down' | 'stable' | 'random';
    seasonality?: boolean;
    noise?: number;
}
export declare const TimeSeriesOptionsSchema: z.ZodObject<{
    count: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    schema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        json: "json";
        csv: "csv";
        array: "array";
    }>>>;
    seed: z.ZodOptional<z.ZodUnion<readonly [z.ZodString, z.ZodNumber]>>;
    constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    startDate: z.ZodOptional<z.ZodUnion<readonly [z.ZodDate, z.ZodString]>>;
    endDate: z.ZodOptional<z.ZodUnion<readonly [z.ZodDate, z.ZodString]>>;
    interval: z.ZodDefault<z.ZodOptional<z.ZodString>>;
    metrics: z.ZodOptional<z.ZodArray<z.ZodString>>;
    trend: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        up: "up";
        down: "down";
        stable: "stable";
        random: "random";
    }>>>;
    seasonality: z.ZodDefault<z.ZodOptional<z.ZodBoolean>>;
    noise: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
}, z.core.$strip>;
export interface EventOptions extends GeneratorOptions {
    eventTypes?: string[];
    distribution?: 'uniform' | 'poisson' | 'normal';
    timeRange?: {
        start: Date | string;
        end: Date | string;
    };
    userCount?: number;
}
export declare const EventOptionsSchema: z.ZodObject<{
    count: z.ZodDefault<z.ZodOptional<z.ZodNumber>>;
    schema: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    format: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        json: "json";
        csv: "csv";
        array: "array";
    }>>>;
    seed: z.ZodOptional<z.ZodUnion<readonly [z.ZodString, z.ZodNumber]>>;
    constraints: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    eventTypes: z.ZodOptional<z.ZodArray<z.ZodString>>;
    distribution: z.ZodDefault<z.ZodOptional<z.ZodEnum<{
        uniform: "uniform";
        poisson: "poisson";
        normal: "normal";
    }>>>;
    timeRange: z.ZodOptional<z.ZodObject<{
        start: z.ZodUnion<readonly [z.ZodDate, z.ZodString]>;
        end: z.ZodUnion<readonly [z.ZodDate, z.ZodString]>;
    }, z.core.$strip>>;
    userCount: z.ZodOptional<z.ZodNumber>;
}, z.core.$strip>;
export interface GenerationResult<T = JsonValue> {
    data: T[];
    metadata: {
        count: number;
        generatedAt: Date;
        provider: ModelProvider;
        model: string;
        cached: boolean;
        duration: number;
    };
}
export declare class SynthError extends Error {
    code: string;
    details?: unknown | undefined;
    constructor(message: string, code: string, details?: unknown | undefined);
}
export declare class ValidationError extends SynthError {
    constructor(message: string, details?: unknown);
}
export declare class APIError extends SynthError {
    constructor(message: string, details?: unknown);
}
export declare class CacheError extends SynthError {
    constructor(message: string, details?: unknown);
}
export interface ModelRoute {
    provider: ModelProvider;
    model: string;
    priority: number;
    capabilities: string[];
}
export interface StreamChunk<T = JsonValue> {
    type: 'data' | 'metadata' | 'error' | 'complete';
    data?: T;
    metadata?: Record<string, unknown>;
    error?: Error;
}
export type StreamCallback<T = JsonValue> = (chunk: StreamChunk<T>) => void | Promise<void>;
//# sourceMappingURL=types.d.ts.map