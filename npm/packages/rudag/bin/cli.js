#!/usr/bin/env node

/**
 * rudag CLI - Command-line interface for DAG operations
 */

const { RuDag, DagOperator, AttentionMechanism, MemoryStorage } = require('../dist/index.js');
const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
const command = args[0];

const help = `
rudag - Self-learning DAG query optimization CLI

Usage: rudag <command> [options]

Commands:
  create <name>             Create a new DAG
  load <file>               Load DAG from file
  info <file>               Show DAG information
  topo <file>               Print topological sort
  critical <file>           Find critical path
  attention <file> [type]   Compute attention scores (type: topo|critical|uniform)
  convert <in> <out>        Convert between JSON and binary formats
  help                      Show this help message

Examples:
  rudag create my-query > my-query.dag
  rudag info my-query.dag
  rudag critical my-query.dag
  rudag attention my-query.dag critical

Options:
  --json                    Output in JSON format
  --verbose                 Verbose output
`;

async function main() {
  if (!command || command === 'help' || command === '--help') {
    console.log(help);
    process.exit(0);
  }

  const isJson = args.includes('--json');
  const verbose = args.includes('--verbose');

  try {
    switch (command) {
      case 'create': {
        const name = args[1] || 'untitled';
        const dag = new RuDag({ name, storage: null, autoSave: false });
        await dag.init();

        // Create a simple example DAG
        const scan = dag.addNode(DagOperator.SCAN, 10.0);
        const filter = dag.addNode(DagOperator.FILTER, 2.0);
        const project = dag.addNode(DagOperator.PROJECT, 1.0);

        dag.addEdge(scan, filter);
        dag.addEdge(filter, project);

        if (isJson) {
          console.log(dag.toJSON());
        } else {
          const bytes = dag.toBytes();
          process.stdout.write(Buffer.from(bytes));
        }
        break;
      }

      case 'load': {
        const file = args[1];
        if (!file) {
          console.error('Error: No file specified');
          process.exit(1);
        }

        const data = fs.readFileSync(file);
        let dag;

        if (file.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        console.log(`Loaded DAG with ${dag.nodeCount} nodes and ${dag.edgeCount} edges`);
        break;
      }

      case 'info': {
        const file = args[1];
        if (!file) {
          console.error('Error: No file specified');
          process.exit(1);
        }

        const data = fs.readFileSync(file);
        let dag;

        if (file.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        const info = {
          file,
          nodes: dag.nodeCount,
          edges: dag.edgeCount,
          criticalPath: dag.criticalPath(),
        };

        if (isJson) {
          console.log(JSON.stringify(info, null, 2));
        } else {
          console.log(`File: ${info.file}`);
          console.log(`Nodes: ${info.nodes}`);
          console.log(`Edges: ${info.edges}`);
          console.log(`Critical Path: ${info.criticalPath.path.join(' -> ')}`);
          console.log(`Total Cost: ${info.criticalPath.cost}`);
        }
        break;
      }

      case 'topo': {
        const file = args[1];
        if (!file) {
          console.error('Error: No file specified');
          process.exit(1);
        }

        const data = fs.readFileSync(file);
        let dag;

        if (file.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        const topo = dag.topoSort();

        if (isJson) {
          console.log(JSON.stringify(topo));
        } else {
          console.log('Topological order:', topo.join(' -> '));
        }
        break;
      }

      case 'critical': {
        const file = args[1];
        if (!file) {
          console.error('Error: No file specified');
          process.exit(1);
        }

        const data = fs.readFileSync(file);
        let dag;

        if (file.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        const result = dag.criticalPath();

        if (isJson) {
          console.log(JSON.stringify(result));
        } else {
          console.log('Critical Path:', result.path.join(' -> '));
          console.log('Total Cost:', result.cost);
        }
        break;
      }

      case 'attention': {
        const file = args[1];
        const type = args[2] || 'critical';

        if (!file) {
          console.error('Error: No file specified');
          process.exit(1);
        }

        const data = fs.readFileSync(file);
        let dag;

        if (file.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        let mechanism;
        switch (type) {
          case 'topo':
          case 'topological':
            mechanism = AttentionMechanism.TOPOLOGICAL;
            break;
          case 'critical':
          case 'critical_path':
            mechanism = AttentionMechanism.CRITICAL_PATH;
            break;
          case 'uniform':
            mechanism = AttentionMechanism.UNIFORM;
            break;
          default:
            console.error(`Unknown attention type: ${type}`);
            process.exit(1);
        }

        const scores = dag.attention(mechanism);

        if (isJson) {
          console.log(JSON.stringify({ type, scores }));
        } else {
          console.log(`Attention type: ${type}`);
          scores.forEach((score, i) => {
            console.log(`  Node ${i}: ${score.toFixed(4)}`);
          });
        }
        break;
      }

      case 'convert': {
        const inFile = args[1];
        const outFile = args[2];

        if (!inFile || !outFile) {
          console.error('Error: Input and output files required');
          process.exit(1);
        }

        const data = fs.readFileSync(inFile);
        let dag;

        if (inFile.endsWith('.json')) {
          dag = await RuDag.fromJSON(data.toString(), { storage: null });
        } else {
          dag = await RuDag.fromBytes(new Uint8Array(data), { storage: null });
        }

        if (outFile.endsWith('.json')) {
          fs.writeFileSync(outFile, dag.toJSON());
        } else {
          fs.writeFileSync(outFile, Buffer.from(dag.toBytes()));
        }

        console.log(`Converted ${inFile} -> ${outFile}`);
        break;
      }

      default:
        console.error(`Unknown command: ${command}`);
        console.log('Run "rudag help" for usage information');
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error.message);
    if (verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
