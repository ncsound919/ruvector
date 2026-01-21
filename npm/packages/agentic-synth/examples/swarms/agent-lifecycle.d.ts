/**
 * Agent Lifecycle Management Examples
 *
 * Demonstrates agent lifecycle patterns including spawning/termination,
 * state synchronization, health checks, recovery patterns, and
 * version migration for dynamic agent systems.
 *
 * Integrates with:
 * - claude-flow: Agent lifecycle hooks and state management
 * - ruv-swarm: Dynamic agent spawning and coordination
 * - Kubernetes: Container orchestration patterns
 */
/**
 * Generate agent spawning and termination lifecycle events
 */
export declare function agentSpawningTermination(): Promise<{
    lifecycleEvents: any;
    spawnStrategies: any;
    resourcePool: any;
}>;
/**
 * Generate state synchronization data for distributed agents
 */
export declare function stateSynchronization(): Promise<{
    stateSnapshots: any;
    syncEvents: any;
    consistencyChecks: any;
    syncTopology: any;
}>;
/**
 * Generate health check and monitoring data
 */
export declare function healthCheckScenarios(): Promise<{
    healthChecks: any;
    healthConfigs: any;
    healthTimeSeries: any;
    healingActions: any;
}>;
/**
 * Generate failure recovery pattern data
 */
export declare function recoveryPatterns(): Promise<{
    failures: any;
    recoveryStrategies: any;
    circuitBreakers: any;
    backupOperations: any;
}>;
/**
 * Generate agent version migration data
 */
export declare function versionMigration(): Promise<{
    versions: any;
    migrations: any;
    canaryDeployments: any;
    rollbacks: any;
}>;
export declare function runAllLifecycleExamples(): Promise<void>;
//# sourceMappingURL=agent-lifecycle.d.ts.map