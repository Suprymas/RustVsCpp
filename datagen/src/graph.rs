//! Data for the graph (breadth-first search) task.
//!
//! The graph is generated with the R-MAT algorithm using the parameters of
//! the Graph500 benchmark, a = 0.57; b = c = 0.19; d = 0.05, and an edge
//! factor of 16 (16 * n edges are generated, as in Graph500). Such a
//! generator yields a skewed vertex degree distribution - and that load
//! imbalance is exactly what this task studies.
//!
//! Processing steps:
//!   1. generate n * degree edges (n = 2^scale);
//!   2. remove self-loops (u == v);
//!   3. randomly permute the vertex numbers (as in Graph500), so that the
//!      generation order does not create artificial memory locality;
//!   4. symmetrise the graph (add the reverse edges);
//!   5. remove duplicate edges;
//!   6. keep the largest connected component, renumber the vertices;
//!   7. write in CSR format together with eight fixed search source
//!      vertices.
//!
//! File format (all numbers little-endian):
//!   bytes  0..4    magic "RCSR"
//!   bytes  4..8    format version (u32) = 1
//!   bytes  8..16   number of vertices n (u64)
//!   bytes 16..24   number of directed entries m (u64)
//!   bytes 24..32   number of search sources s (u64)
//!   then           offsets array (n+1) x u64
//!   then           neighbours array m x u32
//!   then           search source vertices s x u32

use crate::rng::Xoshiro256pp;
use crate::util::{human_bytes, sha256_file, Manifest};
use std::fs::File;
use std::io::{BufWriter, Result as IoResult, Write};
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"RCSR";
pub const VERSION: u32 = 1;
#[allow(dead_code)]
pub const HEADER_BYTES: u64 = 32;

/// R-MAT parameters. d is derived as 1 - a - b - c.
pub const RMAT_A: f64 = 0.57;
pub const RMAT_B: f64 = 0.19;
pub const RMAT_C: f64 = 0.19;

pub const NUM_SOURCES: usize = 8;

pub struct Csr {
    pub offsets: Vec<u64>,
    pub neighbors: Vec<u32>,
}

impl Csr {
    pub fn num_vertices(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn num_entries(&self) -> usize {
        self.neighbors.len()
    }
    #[inline]
    pub fn degree(&self, v: usize) -> usize {
        (self.offsets[v + 1] - self.offsets[v]) as usize
    }
    #[inline]
    pub fn neighbors_of(&self, v: usize) -> &[u32] {
        &self.neighbors[self.offsets[v] as usize..self.offsets[v + 1] as usize]
    }
}

/// One R-MAT edge: at each of the `scale` steps a quadrant of the adjacency
/// matrix is chosen and the corresponding vertex bit is set.
#[inline]
fn rmat_edge(rng: &mut Xoshiro256pp, scale: u32) -> (u32, u32) {
    let ab = RMAT_A + RMAT_B;
    let abc = ab + RMAT_C;
    let mut u: u32 = 0;
    let mut v: u32 = 0;
    for i in 0..scale {
        let bit = 1u32 << (scale - 1 - i);
        let r = rng.next_f64();
        if r < RMAT_A {
            // upper left quadrant - bits unchanged
        } else if r < ab {
            v |= bit;
        } else if r < abc {
            u |= bit;
        } else {
            u |= bit;
            v |= bit;
        }
    }
    (u, v)
}

#[inline]
fn pack(u: u32, v: u32) -> u64 {
    ((u as u64) << 32) | v as u64
}

/// Fisher-Yates permutation of the vertex numbers.
fn random_permutation(n: usize, rng: &mut Xoshiro256pp) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.next_bounded(i as u64 + 1) as usize;
        perm.swap(i, j);
    }
    perm
}

/// Converts a sorted, duplicate-free edge list into CSR.
fn build_csr(n: usize, edges: &[u64]) -> Csr {
    let mut offsets = vec![0u64; n + 1];
    for e in edges {
        offsets[(e >> 32) as usize + 1] += 1;
    }
    for i in 0..n {
        offsets[i + 1] += offsets[i];
    }
    let mut neighbors = vec![0u32; edges.len()];
    // The edges are already sorted by (u, v), so it is enough to fill the
    // neighbours array sequentially - it stays sorted within each row.
    for (i, e) in edges.iter().enumerate() {
        neighbors[i] = (*e & 0xFFFF_FFFF) as u32;
    }
    Csr { offsets, neighbors }
}

/// Finds the largest connected component and returns its vertex mark array.
fn largest_component(csr: &Csr) -> (Vec<bool>, usize) {
    let n = csr.num_vertices();
    let mut comp = vec![u32::MAX; n];
    let mut queue: Vec<u32> = Vec::new();
    let mut best_id: u32 = u32::MAX;
    let mut best_size: usize = 0;
    let mut next_id: u32 = 0;

    for start in 0..n {
        if comp[start] != u32::MAX {
            continue;
        }
        let id = next_id;
        next_id += 1;
        comp[start] = id;
        queue.clear();
        queue.push(start as u32);
        let mut head = 0usize;
        let mut size = 0usize;
        while head < queue.len() {
            let v = queue[head] as usize;
            head += 1;
            size += 1;
            for &w in csr.neighbors_of(v) {
                if comp[w as usize] == u32::MAX {
                    comp[w as usize] = id;
                    queue.push(w);
                }
            }
        }
        if size > best_size {
            best_size = size;
            best_id = id;
        }
    }

    let keep: Vec<bool> = comp.iter().map(|&c| c == best_id).collect();
    (keep, best_size)
}

/// Keeps only the marked vertices and renumbers them 0..k in increasing
/// order of the old numbers.
fn extract_subgraph(csr: &Csr, keep: &[bool], kept_count: usize) -> Csr {
    let n = csr.num_vertices();
    let mut map = vec![u32::MAX; n];
    let mut next: u32 = 0;
    for v in 0..n {
        if keep[v] {
            map[v] = next;
            next += 1;
        }
    }

    let mut offsets = Vec::with_capacity(kept_count + 1);
    offsets.push(0u64);
    let mut neighbors: Vec<u32> = Vec::with_capacity(csr.num_entries());
    for v in 0..n {
        if !keep[v] {
            continue;
        }
        for &w in csr.neighbors_of(v) {
            // In a connected component all neighbours belong to the same component.
            debug_assert!(keep[w as usize]);
            neighbors.push(map[w as usize]);
        }
        offsets.push(neighbors.len() as u64);
    }
    neighbors.shrink_to_fit();
    Csr { offsets, neighbors }
}

fn pick_sources(n: usize, rng: &mut Xoshiro256pp) -> Vec<u32> {
    let mut chosen: Vec<u32> = Vec::with_capacity(NUM_SOURCES);
    while chosen.len() < NUM_SOURCES.min(n) {
        let v = rng.next_bounded(n as u64) as u32;
        if !chosen.contains(&v) {
            chosen.push(v);
        }
    }
    chosen.sort_unstable();
    chosen
}

fn write_csr(path: &Path, csr: &Csr, sources: &[u32]) -> IoResult<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::with_capacity(1 << 20, f);

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(csr.num_vertices() as u64).to_le_bytes())?;
    w.write_all(&(csr.num_entries() as u64).to_le_bytes())?;
    w.write_all(&(sources.len() as u64).to_le_bytes())?;

    let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
    for o in &csr.offsets {
        buf.extend_from_slice(&o.to_le_bytes());
        if buf.len() >= (1 << 20) {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    buf.clear();

    for v in &csr.neighbors {
        buf.extend_from_slice(&v.to_le_bytes());
        if buf.len() >= (1 << 20) {
            w.write_all(&buf)?;
            buf.clear();
        }
    }
    w.write_all(&buf)?;
    buf.clear();

    for s in sources {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    w.write_all(&buf)?;
    w.flush()?;
    Ok(())
}

/// Generates one graph and returns it together with the search sources.
pub fn build_graph(scale: u32, degree: u64, seed: u64, permute: bool) -> (Csr, Vec<u32>, GraphStats) {
    assert!(scale >= 2 && scale <= 31, "scale must be in the range 2..31");
    let n = 1usize << scale;
    let m_gen = n as u64 * degree;

    let mut rng = Xoshiro256pp::stream(seed, &format!("graph/{}/edges", scale));
    let mut edges: Vec<u64> = Vec::with_capacity(m_gen as usize);
    let mut self_loops: u64 = 0;
    for _ in 0..m_gen {
        let (u, v) = rmat_edge(&mut rng, scale);
        if u == v {
            self_loops += 1;
            continue;
        }
        edges.push(pack(u, v));
    }

    if permute {
        let mut prng = Xoshiro256pp::stream(seed, &format!("graph/{}/permutation", scale));
        let perm = random_permutation(n, &mut prng);
        for e in edges.iter_mut() {
            let u = perm[(*e >> 32) as usize];
            let v = perm[(*e & 0xFFFF_FFFF) as usize];
            *e = pack(u, v);
        }
    }

    // Symmetrisation.
    let directed_before = edges.len() as u64;
    edges.reserve(edges.len());
    for i in 0..directed_before as usize {
        let e = edges[i];
        edges.push(pack((e & 0xFFFF_FFFF) as u32, (e >> 32) as u32));
    }

    edges.sort_unstable();
    edges.dedup();
    edges.shrink_to_fit();

    let csr = build_csr(n, &edges);
    drop(edges);

    let (keep, lcc_size) = largest_component(&csr);
    let lcc = extract_subgraph(&csr, &keep, lcc_size);
    let isolated = (0..n).filter(|&v| csr.degree(v) == 0).count();
    drop(csr);

    let mut srng = Xoshiro256pp::stream(seed, &format!("graph/{}/sources", scale));
    let sources = pick_sources(lcc.num_vertices(), &mut srng);

    let max_degree = (0..lcc.num_vertices())
        .map(|v| lcc.degree(v))
        .max()
        .unwrap_or(0);

    let stats = GraphStats {
        scale,
        generated_edges: m_gen,
        self_loops,
        vertices_total: n,
        vertices_isolated: isolated,
        lcc_vertices: lcc.num_vertices(),
        lcc_entries: lcc.num_entries(),
        max_degree,
        permuted: permute,
    };

    (lcc, sources, stats)
}

pub struct GraphStats {
    pub scale: u32,
    pub generated_edges: u64,
    pub self_loops: u64,
    pub vertices_total: usize,
    pub vertices_isolated: usize,
    pub lcc_vertices: usize,
    pub lcc_entries: usize,
    pub max_degree: usize,
    pub permuted: bool,
}

pub fn generate(
    scales: &[u32],
    degree: u64,
    out_dir: &Path,
    seed: u64,
    permute: bool,
    manifest: &mut Manifest,
) -> IoResult<()> {
    manifest.section("Graph data");
    manifest.kv(
        "Algorithm",
        &format!(
            "R-MAT, a = {:.2}, b = {:.2}, c = {:.2}, d = {:.2}",
            RMAT_A,
            RMAT_B,
            RMAT_C,
            1.0 - RMAT_A - RMAT_B - RMAT_C
        ),
    );
    manifest.kv("Edge factor (edgefactor, in the Graph500 sense)", &degree.to_string());
    manifest.kv(
        "Vertex number permutation",
        if permute { "applied" } else { "not applied" },
    );
    manifest.kv(
        "Format",
        "RCSR v1: magic(4) + version(u32) + n(u64) + m(u64) + s(u64) + offsets + neighbours + sources",
    );

    for &scale in scales {
        let (csr, sources, st) = build_graph(scale, degree, seed, permute);
        let name = format!("graph_s{}.csr", scale);
        let path = out_dir.join(&name);
        write_csr(&path, &csr, &sources)?;

        let size = path.metadata()?.len();
        let sha = sha256_file(&path)?;
        let avg_deg = st.lcc_entries as f64 / st.lcc_vertices as f64;

        println!(
            "  {}  {}  n={} m={} mean.degree={:.2} max.degree={}",
            name,
            human_bytes(size),
            st.lcc_vertices,
            st.lcc_entries,
            avg_deg,
            st.max_degree
        );
        manifest.file_entry(&path, size, &sha, "");
        manifest.kv(
            &format!("  scale {} statistics", scale),
            &format!(
                "edges generated {}, self-loops removed {}, vertices total {}, isolated {}, \
                 largest component {} vertices ({:.2} %), directed entries {}, \
                 mean degree {:.2}, max degree {}",
                st.generated_edges,
                st.self_loops,
                st.vertices_total,
                st.vertices_isolated,
                st.lcc_vertices,
                100.0 * st.lcc_vertices as f64 / st.vertices_total as f64,
                st.lcc_entries,
                avg_deg,
                st.max_degree
            ),
        );
        manifest.kv(
            &format!("  scale {} search sources", scale),
            &format!("{:?}", sources),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_is_symmetric_and_connected() {
        let (csr, sources, st) = build_graph(10, 16, 20260101, true);
        assert!(st.lcc_vertices > 0);
        assert_eq!(sources.len(), NUM_SOURCES);

        // Symmetry: for every edge (u, v) there must be an edge (v, u).
        for u in 0..csr.num_vertices() {
            for &v in csr.neighbors_of(u) {
                assert!(
                    csr.neighbors_of(v as usize).contains(&(u as u32)),
                    "missing reverse edge {}->{}",
                    v,
                    u
                );
            }
        }

        // No self-loops and no duplicates.
        for u in 0..csr.num_vertices() {
            let ns = csr.neighbors_of(u);
            assert!(!ns.contains(&(u as u32)), "self-loop found at vertex {}", u);
            assert!(ns.windows(2).all(|w| w[0] < w[1]), "neighbours not sorted/duplicated");
        }

        // All vertices are reachable from any source.
        let n = csr.num_vertices();
        let mut seen = vec![false; n];
        let mut q = vec![0u32];
        seen[0] = true;
        let mut head = 0;
        while head < q.len() {
            let v = q[head] as usize;
            head += 1;
            for &w in csr.neighbors_of(v) {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    q.push(w);
                }
            }
        }
        assert_eq!(q.len(), n, "graph is not connected");
    }

    #[test]
    fn generation_is_deterministic() {
        let (a, sa, _) = build_graph(9, 8, 42, true);
        let (b, sb, _) = build_graph(9, 8, 42, true);
        assert_eq!(a.offsets, b.offsets);
        assert_eq!(a.neighbors, b.neighbors);
        assert_eq!(sa, sb);
    }
}
