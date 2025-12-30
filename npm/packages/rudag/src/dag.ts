/**
 * High-level DAG API with WASM acceleration
 * Provides a TypeScript-friendly interface to the WASM DAG implementation
 */

import { createStorage, DagStorage, MemoryStorage, StoredDag } from './storage';

// WASM module type definitions
interface WasmDagModule {
  WasmDag: {
    new(): WasmDagInstance;
    from_bytes(data: Uint8Array): WasmDagInstance;
    from_json(json: string): WasmDagInstance;
  };
}

interface WasmDagInstance {
  add_node(op: number, cost: number): number;
  add_edge(from: number, to: number): boolean;
  node_count(): number;
  edge_count(): number;
  topo_sort(): Uint32Array;
  critical_path(): unknown;
  attention(mechanism: number): Float32Array;
  to_bytes(): Uint8Array;
  to_json(): string;
  free(): void;
}

/**
 * Operator types for DAG nodes
 */
export enum DagOperator {
  SCAN = 0,
  FILTER = 1,
  PROJECT = 2,
  JOIN = 3,
  AGGREGATE = 4,
  SORT = 5,
  LIMIT = 6,
  UNION = 7,
  CUSTOM = 255,
}

/**
 * Attention mechanism types
 */
export enum AttentionMechanism {
  TOPOLOGICAL = 0,
  CRITICAL_PATH = 1,
  UNIFORM = 2,
}

/**
 * Node representation
 */
export interface DagNode {
  id: number;
  operator: DagOperator | number;
  cost: number;
  metadata?: Record<string, unknown>;
}

/**
 * Edge representation
 */
export interface DagEdge {
  from: number;
  to: number;
}

/**
 * Critical path result
 */
export interface CriticalPath {
  path: number[];
  cost: number;
}

/**
 * DAG configuration options
 */
export interface RuDagOptions {
  id?: string;
  name?: string;
  storage?: DagStorage | MemoryStorage | null;
  autoSave?: boolean;
}

let wasmModule: WasmDagModule | null = null;

/**
 * Initialize WASM module
 */
async function initWasm(): Promise<WasmDagModule> {
  if (wasmModule) return wasmModule;

  try {
    // Try browser bundler version first
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mod = await import('../pkg/ruvector_dag_wasm.js') as any;
    if (typeof mod.default === 'function') {
      await mod.default();
    }
    wasmModule = mod as WasmDagModule;
    return wasmModule;
  } catch {
    try {
      // Fallback to Node.js version
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mod = await import('../pkg-node/ruvector_dag_wasm.js') as any;
      wasmModule = mod as WasmDagModule;
      return wasmModule;
    } catch (e) {
      throw new Error(`Failed to load WASM module: ${e}`);
    }
  }
}

/**
 * RuDag - High-performance DAG with WASM acceleration and persistence
 */
export class RuDag {
  private wasm: WasmDagInstance | null = null;
  private nodes: Map<number, DagNode> = new Map();
  private storage: DagStorage | MemoryStorage | null;
  private id: string;
  private name?: string;
  private autoSave: boolean;
  private initialized = false;

  constructor(options: RuDagOptions = {}) {
    this.id = options.id || `dag-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    this.name = options.name;
    this.storage = options.storage === undefined ? createStorage() : options.storage;
    this.autoSave = options.autoSave ?? true;
  }

  /**
   * Initialize the DAG
   */
  async init(): Promise<this> {
    if (this.initialized) return this;

    const mod = await initWasm();
    this.wasm = new mod.WasmDag();

    if (this.storage) {
      await this.storage.init();
    }

    this.initialized = true;
    return this;
  }

  /**
   * Ensure DAG is initialized
   */
  private ensureInit(): WasmDagInstance {
    if (!this.wasm) {
      throw new Error('DAG not initialized. Call init() first.');
    }
    return this.wasm;
  }

  /**
   * Add a node to the DAG
   */
  addNode(operator: DagOperator | number, cost: number, metadata?: Record<string, unknown>): number {
    const wasm = this.ensureInit();
    const id = wasm.add_node(operator, cost);

    this.nodes.set(id, {
      id,
      operator,
      cost,
      metadata,
    });

    if (this.autoSave) {
      this.save().catch(() => {}); // Background save
    }

    return id;
  }

  /**
   * Add an edge between nodes
   */
  addEdge(from: number, to: number): boolean {
    const wasm = this.ensureInit();
    const success = wasm.add_edge(from, to);

    if (success && this.autoSave) {
      this.save().catch(() => {}); // Background save
    }

    return success;
  }

  /**
   * Get node count
   */
  get nodeCount(): number {
    return this.ensureInit().node_count();
  }

  /**
   * Get edge count
   */
  get edgeCount(): number {
    return this.ensureInit().edge_count();
  }

  /**
   * Get topological sort
   */
  topoSort(): number[] {
    const result = this.ensureInit().topo_sort();
    return Array.from(result);
  }

  /**
   * Find critical path
   */
  criticalPath(): CriticalPath {
    const result = this.ensureInit().critical_path();

    if (typeof result === 'string') {
      return JSON.parse(result);
    }
    return result as CriticalPath;
  }

  /**
   * Compute attention scores
   */
  attention(mechanism: AttentionMechanism = AttentionMechanism.CRITICAL_PATH): number[] {
    const result = this.ensureInit().attention(mechanism);
    return Array.from(result);
  }

  /**
   * Get node by ID
   */
  getNode(id: number): DagNode | undefined {
    return this.nodes.get(id);
  }

  /**
   * Get all nodes
   */
  getNodes(): DagNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Serialize to bytes
   */
  toBytes(): Uint8Array {
    return this.ensureInit().to_bytes();
  }

  /**
   * Serialize to JSON
   */
  toJSON(): string {
    return this.ensureInit().to_json();
  }

  /**
   * Save DAG to storage
   */
  async save(): Promise<StoredDag | null> {
    if (!this.storage) return null;

    const data = this.toBytes();
    return this.storage.save(this.id, data, {
      name: this.name,
      metadata: {
        nodeCount: this.nodeCount,
        edgeCount: this.edgeCount,
        nodes: Object.fromEntries(this.nodes),
      },
    });
  }

  /**
   * Load DAG from storage by ID
   */
  static async load(id: string, storage?: DagStorage | MemoryStorage): Promise<RuDag | null> {
    const store = storage || createStorage();
    await store.init();

    const record = await store.get(id);
    if (!record) return null;

    return RuDag.fromBytes(record.data, {
      id: record.id,
      name: record.name,
      storage: store,
    });
  }

  /**
   * Create DAG from bytes
   */
  static async fromBytes(data: Uint8Array, options: RuDagOptions = {}): Promise<RuDag> {
    const mod = await initWasm();
    const dag = new RuDag(options);
    dag.wasm = mod.WasmDag.from_bytes(data);
    dag.initialized = true;

    if (dag.storage) {
      await dag.storage.init();
    }

    return dag;
  }

  /**
   * Create DAG from JSON
   */
  static async fromJSON(json: string, options: RuDagOptions = {}): Promise<RuDag> {
    const mod = await initWasm();
    const dag = new RuDag(options);
    dag.wasm = mod.WasmDag.from_json(json);
    dag.initialized = true;

    if (dag.storage) {
      await dag.storage.init();
    }

    return dag;
  }

  /**
   * List all stored DAGs
   */
  static async listStored(storage?: DagStorage | MemoryStorage): Promise<StoredDag[]> {
    const store = storage || createStorage();
    await store.init();
    return store.list();
  }

  /**
   * Delete a stored DAG
   */
  static async deleteStored(id: string, storage?: DagStorage | MemoryStorage): Promise<boolean> {
    const store = storage || createStorage();
    await store.init();
    return store.delete(id);
  }

  /**
   * Get storage statistics
   */
  static async storageStats(storage?: DagStorage | MemoryStorage): Promise<{ count: number; totalSize: number }> {
    const store = storage || createStorage();
    await store.init();
    return store.stats();
  }

  /**
   * Get DAG ID
   */
  getId(): string {
    return this.id;
  }

  /**
   * Get DAG name
   */
  getName(): string | undefined {
    return this.name;
  }

  /**
   * Set DAG name
   */
  setName(name: string): void {
    this.name = name;
    if (this.autoSave) {
      this.save().catch(() => {});
    }
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    if (this.wasm) {
      this.wasm.free();
      this.wasm = null;
    }
    if (this.storage) {
      this.storage.close();
    }
    this.initialized = false;
  }
}
