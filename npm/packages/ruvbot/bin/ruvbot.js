#!/usr/bin/env node

/**
 * RuvBot CLI Entry Point
 *
 * Usage:
 *   npx @ruvector/ruvbot <command> [options]
 *   ruvbot <command> [options]
 *
 * Commands:
 *   start     Start the RuvBot server
 *   init      Initialize RuvBot in current directory
 *   doctor    Run diagnostics and health checks
 *   config    Manage configuration
 *   memory    Memory management commands
 *   security  Security scanning and audit
 *   plugins   Plugin management
 *   agent     Agent management
 *   status    Show bot status
 */

import 'dotenv/config';
import { main } from '../dist/esm/cli/index.js';

main().catch((error) => {
  console.error('Fatal error:', error.message);
  process.exit(1);
});
