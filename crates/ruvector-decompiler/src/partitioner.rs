//! MinCut-based module boundary detection.
//!
//! Uses `ruvector-mincut`'s `GraphPartitioner` to split the reference graph
//! into partitions, each representing a reconstructed module.

use crate::error::{DecompilerError, Result};
use crate::graph::ReferenceGraph;
use crate::types::{Declaration, Module};

use ruvector_mincut::GraphPartitioner;

/// Partition the reference graph into modules using MinCut bisection.
///
/// If `target_modules` is `None`, the partition count is estimated from
/// the graph structure (heuristic: one module per 3--5 loosely connected
/// declarations, minimum 2).
pub fn partition_modules(
    graph: &ReferenceGraph,
    target_modules: Option<usize>,
) -> Result<Vec<Module>> {
    let n = graph.node_count();
    if n == 0 {
        return Err(DecompilerError::PartitioningFailed(
            "no declarations to partition".to_string(),
        ));
    }

    // Determine target partition count.
    let target = target_modules.unwrap_or_else(|| estimate_module_count(graph));
    let target = target.clamp(1, n);

    if target == 1 || n <= 2 {
        // Everything in one module.
        return Ok(vec![build_module(
            0,
            &graph.declarations,
            &graph.declarations,
        )]);
    }

    // Use MinCut GraphPartitioner for recursive bisection.
    let partitioner = GraphPartitioner::new(graph.graph.clone(), target);
    let partitions = partitioner.partition();

    // Track which declarations were assigned by the partitioner.
    let mut assigned: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut modules = Vec::new();
    let mut mod_idx = 0;

    for partition in &partitions {
        if partition.is_empty() {
            continue;
        }

        let mut decls: Vec<Declaration> = Vec::new();
        for &vid in partition {
            if let Some(idx) = graph.vertex_to_decl.get(&vid) {
                if let Some(decl) = graph.declarations.get(*idx) {
                    decls.push(decl.clone());
                    assigned.insert(*idx);
                }
            }
        }

        if !decls.is_empty() {
            modules.push(build_module(mod_idx, &decls, &graph.declarations));
            mod_idx += 1;
        }
    }

    // Collect declarations not assigned by the partitioner (isolated nodes
    // with no edges in the reference graph). Distribute them round-robin
    // across existing modules, or create a new module if none exist.
    let orphans: Vec<Declaration> = graph
        .declarations
        .iter()
        .enumerate()
        .filter(|(i, _)| !assigned.contains(i))
        .map(|(_, d)| d.clone())
        .collect();

    if !orphans.is_empty() {
        if modules.is_empty() {
            // No modules at all: put everything in one.
            modules.push(build_module(0, &orphans, &graph.declarations));
        } else {
            // Distribute orphans by proximity (byte position).
            for orphan in &orphans {
                let best_module = modules
                    .iter_mut()
                    .min_by_key(|m| {
                        let mid = (m.byte_range.0 + m.byte_range.1) / 2;
                        let orphan_mid = (orphan.byte_range.0 + orphan.byte_range.1) / 2;
                        (mid as i64 - orphan_mid as i64).unsigned_abs()
                    })
                    .unwrap();
                best_module.declarations.push(orphan.clone());
                // Update byte range.
                best_module.byte_range.0 =
                    best_module.byte_range.0.min(orphan.byte_range.0);
                best_module.byte_range.1 =
                    best_module.byte_range.1.max(orphan.byte_range.1);
            }
        }
    }

    // Fall back if everything somehow ended up empty.
    if modules.is_empty() {
        return Ok(vec![build_module(
            0,
            &graph.declarations,
            &graph.declarations,
        )]);
    }

    Ok(modules)
}

/// Build a `Module` from a set of declarations.
fn build_module(
    index: usize,
    decls: &[Declaration],
    _all_decls: &[Declaration],
) -> Module {
    let name = infer_module_name(decls, index);

    // Compute the byte range spanning all declarations in this module.
    let start = decls.iter().map(|d| d.byte_range.0).min().unwrap_or(0);
    let end = decls.iter().map(|d| d.byte_range.1).max().unwrap_or(0);

    Module {
        name,
        index,
        declarations: decls.to_vec(),
        source: String::new(), // Filled in by the beautifier later.
        byte_range: (start, end),
    }
}

/// Infer a module name from the dominant string literals and property names.
fn infer_module_name(decls: &[Declaration], fallback_index: usize) -> String {
    // Collect all string literals across declarations in this module.
    let mut candidates: Vec<&str> = Vec::new();

    for decl in decls {
        for s in &decl.string_literals {
            // Prefer short, path-like or keyword-like strings.
            if s.len() >= 2 && s.len() <= 40 && !s.contains(' ') {
                candidates.push(s.as_str());
            }
        }
        for p in &decl.property_accesses {
            candidates.push(p.as_str());
        }
    }

    // Pick the most common non-trivial candidate.
    if !candidates.is_empty() {
        let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for c in &candidates {
            *freq.entry(c).or_insert(0) += 1;
        }
        if let Some((&best, _)) = freq.iter().max_by_key(|(_, &count)| count) {
            return sanitize_module_name(best);
        }
    }

    format!("module_{}", fallback_index)
}

/// Sanitize a string into a valid module name.
fn sanitize_module_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "module".to_string()
    } else {
        cleaned
    }
}

/// Estimate the number of modules based on graph structure.
fn estimate_module_count(graph: &ReferenceGraph) -> usize {
    let n = graph.node_count();
    let e = graph.edge_count();

    if n <= 3 {
        return 1;
    }

    // Heuristic: modules ~ n / avg_degree, clamped to reasonable range.
    let avg_degree = if n > 0 { (2 * e) as f64 / n as f64 } else { 0.0 };

    if avg_degree < 1.0 {
        // Very sparse: likely many independent modules.
        (n / 2).max(2)
    } else {
        // Moderate coupling: fewer modules.
        (n as f64 / (avg_degree + 1.0)).ceil().max(2.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_reference_graph;
    use crate::types::DeclKind;

    fn make_decl(name: &str, refs: &[&str], strings: &[&str]) -> Declaration {
        Declaration {
            name: name.to_string(),
            kind: DeclKind::Var,
            byte_range: (0, 10),
            string_literals: strings.iter().map(|s| s.to_string()).collect(),
            property_accesses: vec![],
            references: refs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_partition_single() {
        let decls = vec![make_decl("a", &[], &["hello"])];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(1)).unwrap();
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_partition_multiple() {
        let decls = vec![
            make_decl("a", &[], &["alpha"]),
            make_decl("b", &["a"], &["beta"]),
            make_decl("c", &[], &["gamma"]),
            make_decl("d", &["c"], &["delta"]),
        ];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(2)).unwrap();
        assert!(!modules.is_empty());
        // Total declarations across all modules should equal 4.
        let total: usize = modules.iter().map(|m| m.declarations.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_module_naming() {
        let decls = vec![make_decl("x", &[], &["auth", "auth", "login"])];
        let name = infer_module_name(&decls, 0);
        assert_eq!(name, "auth");
    }
}
