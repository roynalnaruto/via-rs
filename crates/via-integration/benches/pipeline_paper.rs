//! Paper-scale (n2048 RNS) per-step benchmarks — opt-in, reduced sampling.
//!
//! Placeholder; filled in after the toy suite (`pipeline_toy`) lands.
#![allow(missing_docs)] // criterion_group! generates undocumented public items

use criterion::{Criterion, criterion_group, criterion_main};

fn paper_benches(_c: &mut Criterion) {}

criterion_group!(benches, paper_benches);
criterion_main!(benches);
