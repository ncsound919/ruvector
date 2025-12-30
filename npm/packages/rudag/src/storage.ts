/**
 * IndexedDB-based persistence layer for DAG storage
 * Provides browser-compatible persistent storage for DAGs
 */

const DB_NAME = 'rudag-storage';
const DB_VERSION = 1;
const STORE_NAME = 'dags';

export interface StoredDag {
  id: string;
  name?: string;
  data: Uint8Array;
  createdAt: number;
  updatedAt: number;
  metadata?: Record<string, unknown>;
}

export interface DagStorageOptions {
  dbName?: string;
  version?: number;
}

/**
 * Check if IndexedDB is available (browser environment)
 */
export function isIndexedDBAvailable(): boolean {
  return typeof indexedDB !== 'undefined';
}

/**
 * IndexedDB storage class for DAG persistence
 */
export class DagStorage {
  private dbName: string;
  private version: number;
  private db: IDBDatabase | null = null;

  constructor(options: DagStorageOptions = {}) {
    this.dbName = options.dbName || DB_NAME;
    this.version = options.version || DB_VERSION;
  }

  /**
   * Initialize the database connection
   */
  async init(): Promise<void> {
    if (!isIndexedDBAvailable()) {
      throw new Error('IndexedDB is not available in this environment');
    }

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);

      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'id' });
          store.createIndex('name', 'name', { unique: false });
          store.createIndex('createdAt', 'createdAt', { unique: false });
          store.createIndex('updatedAt', 'updatedAt', { unique: false });
        }
      };
    });
  }

  /**
   * Ensure database is initialized
   */
  private ensureInit(): IDBDatabase {
    if (!this.db) {
      throw new Error('Database not initialized. Call init() first.');
    }
    return this.db;
  }

  /**
   * Save a DAG to storage
   */
  async save(id: string, data: Uint8Array, options: { name?: string; metadata?: Record<string, unknown> } = {}): Promise<StoredDag> {
    const db = this.ensureInit();
    const now = Date.now();

    // Check if exists for update timestamp
    const existing = await this.get(id);

    const record: StoredDag = {
      id,
      name: options.name,
      data,
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      metadata: options.metadata,
    };

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.put(record);

      request.onsuccess = () => resolve(record);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get a DAG from storage
   */
  async get(id: string): Promise<StoredDag | null> {
    const db = this.ensureInit();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(id);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Delete a DAG from storage
   */
  async delete(id: string): Promise<boolean> {
    const db = this.ensureInit();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.delete(id);

      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * List all DAGs in storage
   */
  async list(): Promise<StoredDag[]> {
    const db = this.ensureInit();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Search DAGs by name
   */
  async findByName(name: string): Promise<StoredDag[]> {
    const db = this.ensureInit();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('name');
      const request = index.getAll(name);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Clear all DAGs from storage
   */
  async clear(): Promise<void> {
    const db = this.ensureInit();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction([STORE_NAME], 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get storage statistics
   */
  async stats(): Promise<{ count: number; totalSize: number }> {
    const dags = await this.list();
    const totalSize = dags.reduce((sum, dag) => sum + dag.data.byteLength, 0);
    return { count: dags.length, totalSize };
  }

  /**
   * Close the database connection
   */
  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

/**
 * In-memory storage fallback for Node.js or environments without IndexedDB
 */
export class MemoryStorage {
  private store: Map<string, StoredDag> = new Map();

  async init(): Promise<void> {
    // No-op for memory storage
  }

  async save(id: string, data: Uint8Array, options: { name?: string; metadata?: Record<string, unknown> } = {}): Promise<StoredDag> {
    const now = Date.now();
    const existing = this.store.get(id);

    const record: StoredDag = {
      id,
      name: options.name,
      data,
      createdAt: existing?.createdAt || now,
      updatedAt: now,
      metadata: options.metadata,
    };

    this.store.set(id, record);
    return record;
  }

  async get(id: string): Promise<StoredDag | null> {
    return this.store.get(id) || null;
  }

  async delete(id: string): Promise<boolean> {
    return this.store.delete(id);
  }

  async list(): Promise<StoredDag[]> {
    return Array.from(this.store.values());
  }

  async findByName(name: string): Promise<StoredDag[]> {
    return Array.from(this.store.values()).filter(dag => dag.name === name);
  }

  async clear(): Promise<void> {
    this.store.clear();
  }

  async stats(): Promise<{ count: number; totalSize: number }> {
    const dags = Array.from(this.store.values());
    const totalSize = dags.reduce((sum, dag) => sum + dag.data.byteLength, 0);
    return { count: dags.length, totalSize };
  }

  close(): void {
    // No-op for memory storage
  }
}

/**
 * Create appropriate storage based on environment
 */
export function createStorage(options: DagStorageOptions = {}): DagStorage | MemoryStorage {
  if (isIndexedDBAvailable()) {
    return new DagStorage(options);
  }
  return new MemoryStorage();
}
