/**
 * Distributed Processing Examples
 *
 * Demonstrates distributed computation patterns including map-reduce,
 * worker pools, message queues, event-driven architectures, and
 * saga pattern transactions for multi-agent systems.
 *
 * Integrates with:
 * - claude-flow: Distributed workflow orchestration
 * - Apache Kafka: Event streaming
 * - RabbitMQ: Message queuing
 * - Redis: Distributed caching
 */
/**
 * Generate map-reduce job execution data for distributed processing
 */
export declare function mapReduceJobData(): Promise<any>;
/**
 * Generate worker pool execution data
 */
export declare function workerPoolSimulation(): Promise<{
    poolStates: any;
    workerMetrics: any;
    taskExecutions: any;
}>;
/**
 * Generate message queue data (RabbitMQ, SQS, etc.)
 */
export declare function messageQueueScenarios(): Promise<{
    queueMetrics: any;
    messages: any;
    consumers: any;
}>;
/**
 * Generate event-driven architecture data (Kafka, EventBridge)
 */
export declare function eventDrivenArchitecture(): Promise<{
    events: any;
    handlers: any;
    projections: any;
}>;
/**
 * Generate saga pattern distributed transaction data
 */
export declare function sagaPatternTransactions(): Promise<{
    sagas: any;
    sagaEvents: any;
}>;
/**
 * Generate stream processing pipeline data (Kafka Streams, Flink)
 */
export declare function streamProcessingPipeline(): Promise<{
    pipeline: any;
    metrics: any;
    aggregations: any;
}>;
export declare function runAllDistributedProcessingExamples(): Promise<void>;
//# sourceMappingURL=distributed-processing.d.ts.map