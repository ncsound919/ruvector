/**
 * Node.js-specific entry point with filesystem support
 */

export * from './index';

import { RuDag, MemoryStorage } from './index';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Create a Node.js DAG with memory storage
 */
export async function createNodeDag(name?: string): Promise<RuDag> {
  const storage = new MemoryStorage();
  const dag = new RuDag({ name, storage });
  await dag.init();
  return dag;
}

/**
 * File-based storage for Node.js environments
 */
export class FileDagStorage {
  private basePath: string;

  constructor(basePath: string = '.rudag') {
    this.basePath = basePath;
  }

  async init(): Promise<void> {
    if (!fs.existsSync(this.basePath)) {
      fs.mkdirSync(this.basePath, { recursive: true });
    }
  }

  private getFilePath(id: string): string {
    return path.join(this.basePath, `${id}.dag`);
  }

  private getMetaPath(id: string): string {
    return path.join(this.basePath, `${id}.meta.json`);
  }

  async save(id: string, data: Uint8Array, options: { name?: string; metadata?: Record<string, unknown> } = {}): Promise<void> {
    await this.init();

    fs.writeFileSync(this.getFilePath(id), Buffer.from(data));
    fs.writeFileSync(this.getMetaPath(id), JSON.stringify({
      id,
      name: options.name,
      metadata: options.metadata,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }));
  }

  async load(id: string): Promise<Uint8Array | null> {
    const filePath = this.getFilePath(id);
    if (!fs.existsSync(filePath)) {
      return null;
    }
    return new Uint8Array(fs.readFileSync(filePath));
  }

  async delete(id: string): Promise<boolean> {
    const filePath = this.getFilePath(id);
    const metaPath = this.getMetaPath(id);

    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    if (fs.existsSync(metaPath)) {
      fs.unlinkSync(metaPath);
    }
    return true;
  }

  async list(): Promise<string[]> {
    await this.init();

    const files = fs.readdirSync(this.basePath);
    return files
      .filter(f => f.endsWith('.dag'))
      .map(f => f.replace('.dag', ''));
  }
}

/**
 * Node.js DAG manager with file persistence
 */
export class NodeDagManager {
  private storage: FileDagStorage;

  constructor(basePath?: string) {
    this.storage = new FileDagStorage(basePath);
  }

  async init(): Promise<void> {
    await this.storage.init();
  }

  async createDag(name?: string): Promise<RuDag> {
    const dag = new RuDag({ name, storage: null, autoSave: false });
    await dag.init();
    return dag;
  }

  async saveDag(dag: RuDag): Promise<void> {
    const data = dag.toBytes();
    await this.storage.save(dag.getId(), data, { name: dag.getName() });
  }

  async loadDag(id: string): Promise<RuDag | null> {
    const data = await this.storage.load(id);
    if (!data) return null;
    return RuDag.fromBytes(data, { id });
  }

  async deleteDag(id: string): Promise<boolean> {
    return this.storage.delete(id);
  }

  async listDags(): Promise<string[]> {
    return this.storage.list();
  }
}
