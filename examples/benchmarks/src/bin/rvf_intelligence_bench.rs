//! RVF Intelligence Benchmark Runner
//!
//! Runs head-to-head comparison: Baseline (no learning) vs. RVF-Learning
//! (witness chains + coherence + authority + ReasoningBank).
//!
//! Usage:
//!   cargo run --bin rvf-intelligence-bench -- --episodes 15 --tasks 25 --verbose

use anyhow::Result;
use clap::Parser;
use ruvector_benchmarks::intelligence_metrics::IntelligenceCalculator;
use ruvector_benchmarks::rvf_intelligence_bench::{run_comparison, BenchmarkConfig};

#[derive(Parser, Debug)]
#[command(name = "rvf-intelligence-bench")]
#[command(about = "Benchmark intelligence with and without RVF learning")]
struct Args {
    /// Number of episodes per mode
    #[arg(short, long, default_value = "10")]
    episodes: usize,

    /// Tasks per episode
    #[arg(short, long, default_value = "20")]
    tasks: usize,

    /// Minimum difficulty (1-10)
    #[arg(long, default_value = "1")]
    min_diff: u8,

    /// Maximum difficulty (1-10)
    #[arg(long, default_value = "10")]
    max_diff: u8,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Token budget per episode (RVF mode)
    #[arg(long, default_value = "200000")]
    token_budget: u32,

    /// Tool call budget per episode (RVF mode)
    #[arg(long, default_value = "50")]
    tool_budget: u16,

    /// Verbose per-episode output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!();
    println!("================================================================");
    println!("  RVF Intelligence Benchmark");
    println!("  Measuring cognitive performance: Baseline vs. RVF-Learning");
    println!("================================================================");
    println!();
    println!("  Configuration:");
    println!("    Episodes:       {}", args.episodes);
    println!("    Tasks/episode:  {}", args.tasks);
    println!("    Difficulty:     {}-{}", args.min_diff, args.max_diff);
    println!("    Seed:           {}", args.seed);
    println!("    Token budget:   {}", args.token_budget);
    println!("    Tool budget:    {}", args.tool_budget);
    println!();

    let config = BenchmarkConfig {
        episodes: args.episodes,
        tasks_per_episode: args.tasks,
        min_difficulty: args.min_diff,
        max_difficulty: args.max_diff,
        seed: Some(args.seed),
        token_budget: args.token_budget,
        tool_call_budget: args.tool_budget,
        verbose: args.verbose,
        ..Default::default()
    };

    // Run both modes
    println!("  Phase 1/3: Running baseline (no learning)...");
    if !args.verbose {
        print!("    ");
    }

    let report = run_comparison(&config)?;

    if !args.verbose {
        println!();
    }

    // Print comparison report
    report.print();

    // Also compute full IntelligenceAssessment for each mode
    let calculator = IntelligenceCalculator::default();

    println!("----------------------------------------------------------------");
    println!("  Detailed Intelligence Assessment: Baseline");
    println!("----------------------------------------------------------------");
    let base_assessment = calculator.calculate(&report.baseline.raw_metrics);
    print_compact_assessment(&base_assessment);

    println!();
    println!("----------------------------------------------------------------");
    println!("  Detailed Intelligence Assessment: RVF-Learning");
    println!("----------------------------------------------------------------");
    let rvf_assessment = calculator.calculate(&report.rvf_learning.raw_metrics);
    print_compact_assessment(&rvf_assessment);

    // Final intelligence score comparison
    println!();
    println!("================================================================");
    println!("  Intelligence Score Comparison");
    println!("================================================================");
    println!(
        "  Baseline IQ Score:     {:.1}/100",
        base_assessment.overall_score
    );
    println!(
        "  RVF-Learning IQ Score: {:.1}/100",
        rvf_assessment.overall_score
    );
    println!(
        "  Delta:                 {:+.1}",
        rvf_assessment.overall_score - base_assessment.overall_score
    );
    println!();

    let iq_delta = rvf_assessment.overall_score - base_assessment.overall_score;
    if iq_delta > 5.0 {
        println!("  >> RVF learning loop provides a SIGNIFICANT intelligence boost.");
    } else if iq_delta > 1.0 {
        println!("  >> RVF learning loop provides a MEASURABLE intelligence improvement.");
    } else if iq_delta > 0.0 {
        println!("  >> RVF learning loop provides a MARGINAL intelligence gain.");
    } else {
        println!("  >> Performance is comparable. Increase episodes for stronger signal.");
    }
    println!();

    Ok(())
}

fn print_compact_assessment(
    a: &ruvector_benchmarks::intelligence_metrics::IntelligenceAssessment,
) {
    println!("  Overall Score: {:.1}/100", a.overall_score);
    println!(
        "  Reasoning:     coherence={:.2}, efficiency={:.2}, error_rate={:.2}",
        a.reasoning.logical_coherence,
        a.reasoning.reasoning_efficiency,
        a.reasoning.error_rate,
    );
    println!(
        "  Learning:      sample_eff={:.2}, regret_sub={:.2}, rate={:.2}, gen={:.2}",
        a.learning.sample_efficiency,
        a.learning.regret_sublinearity,
        a.learning.learning_rate,
        a.learning.generalization,
    );
    println!(
        "  Capabilities:  pattern={:.1}, planning={:.1}, adaptation={:.1}",
        a.capabilities.pattern_recognition,
        a.capabilities.planning,
        a.capabilities.adaptation,
    );
    println!(
        "  Meta-cog:      self_correct={:.2}, strategy_adapt={:.2}",
        a.meta_cognition.self_correction_rate, a.meta_cognition.strategy_adaptation,
    );
}
