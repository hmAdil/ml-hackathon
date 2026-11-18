#!/usr/bin/env python3
"""PathSnap Phase 2.3: inspect road-network nodes, arcs, and reference routes."""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

EARTH_RADIUS_M = 6_371_000.0

def load_rows(path: Path, dtype):
    arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def haversine_m(a, b):
    lon1, lat1 = np.radians(a[:, 0]), np.radians(a[:, 1])
    lon2, lat2 = np.radians(b[:, 0]), np.radians(b[:, 1])
    dlat, dlon = lat2-lat1, lon2-lon1
    h = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_RADIUS_M*np.arcsin(np.sqrt(np.clip(h, 0, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data/Student folder',
                    help='Folder containing train/ and student_test/')
    ap.add_argument('--output-dir', default='outputs/phase2_3')
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, breaks = [], []
    all_edge_lengths, all_degrees = [], []
    for partition in ('train', 'student_test'):
        part_dir = root / partition
        if not part_dir.exists():
            print(f'[WARN] Missing partition: {part_dir}')
            continue
        for trip_dir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            trip = trip_dir.name
            nodes_path = trip_dir / f'{trip}.nodes'
            arcs_path = trip_dir / f'{trip}.arcs'
            route_path = trip_dir / f'{trip}.route'
            missing = [p.name for p in (nodes_path, arcs_path, route_path) if not p.exists()]
            if missing:
                print(f'[WARN] {partition}/{trip}: missing {missing}')
                continue
            nodes = load_rows(nodes_path, float)
            arcs = load_rows(arcs_path, np.int64)
            route = np.loadtxt(route_path, dtype=np.int64).reshape(-1)
            if nodes.shape[1] < 2 or arcs.shape[1] != 2:
                raise ValueError(f'Unexpected schema in {partition}/{trip}: nodes={nodes.shape}, arcs={arcs.shape}')
            n, m, r = len(nodes), len(arcs), len(route)
            endpoints_valid = bool(np.all(arcs >= 0) and np.all(arcs < n))
            route_ids_valid = bool(np.all(route >= 0) and np.all(route < m))
            selected = arcs[route] if route_ids_valid else np.empty((0, 2), dtype=np.int64)
            pairs = max(r-1, 0)
            directed_ok = int(np.sum(selected[:-1, 1] == selected[1:, 0])) if len(selected) > 1 else 0
            undirected_ok = int(np.sum(
                (selected[:-1, 1] == selected[1:, 0]) |
                (selected[:-1, 0] == selected[1:, 1])
            )) if len(selected) > 1 else 0
            if route_ids_valid and pairs:
                bad_idx = np.flatnonzero(selected[:-1, 1] != selected[1:, 0])
                for i in bad_idx:
                    breaks.append({
                        'partition': partition, 'trip': trip,
                        'transition_index_zero_based': int(i),
                        'previous_arc_id': int(route[i]),
                        'previous_from_node': int(selected[i, 0]),
                        'previous_to_node': int(selected[i, 1]),
                        'next_arc_id': int(route[i+1]),
                        'next_from_node': int(selected[i+1, 0]),
                        'next_to_node': int(selected[i+1, 1]),
                    })
            # Graph diagnostics
            graph = csr_matrix((np.ones(m, dtype=np.uint8), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
            weak_count, weak_labels = connected_components(graph, directed=True, connection='weak')
            weak_sizes = np.bincount(weak_labels)
            strong_count, _ = connected_components(graph, directed=True, connection='strong')
            out_degree = np.bincount(arcs[:, 0], minlength=n)
            in_degree = np.bincount(arcs[:, 1], minlength=n)
            all_degrees.extend(out_degree.tolist())
            edge_lengths = haversine_m(nodes[arcs[:, 0]], nodes[arcs[:, 1]]) if endpoints_valid else np.array([])
            all_edge_lengths.extend(edge_lengths.tolist())
            unique_arcs = len(np.unique(arcs, axis=0))
            reverse_set = set(map(tuple, arcs.tolist()))
            reverse_count = sum((int(v), int(u)) in reverse_set for u, v in arcs)
            summary = {
                'partition': partition, 'trip': trip,
                'node_count': n, 'directed_arc_count': m, 'reference_route_length': r,
                'arc_endpoints_valid': endpoints_valid, 'route_arc_ids_valid': route_ids_valid,
                'route_transitions': pairs, 'directed_connected_transitions': directed_ok,
                'undirected_connected_transitions': undirected_ok,
                'route_passed_directed': bool(directed_ok == pairs and route_ids_valid),
                'duplicate_directed_arc_rows': int(m - unique_arcs),
                'self_loop_arc_rows': int(np.sum(arcs[:, 0] == arcs[:, 1])),
                'duplicate_node_coordinate_rows': int(n - len(np.unique(nodes[:, :2], axis=0))),
                'weak_component_count': int(weak_count),
                'largest_weak_component_nodes': int(weak_sizes.max()) if len(weak_sizes) else 0,
                'largest_weak_component_pct': float(100*weak_sizes.max()/n) if n else 0.0,
                'strong_component_count': int(strong_count),
                'zero_outdegree_nodes': int(np.sum(out_degree == 0)),
                'zero_indegree_nodes': int(np.sum(in_degree == 0)),
                'max_outdegree': int(out_degree.max()) if n else 0,
                'max_indegree': int(in_degree.max()) if n else 0,
                'reverse_arc_fraction': float(reverse_count/m) if m else 0.0,
                'zero_length_arc_rows': int(np.sum(edge_lengths == 0)),
                'arc_length_median_m': float(np.median(edge_lengths)) if len(edge_lengths) else None,
                'arc_length_p95_m': float(np.quantile(edge_lengths, .95)) if len(edge_lengths) else None,
                'arc_length_p99_m': float(np.quantile(edge_lengths, .99)) if len(edge_lengths) else None,
                'arc_length_max_m': float(np.max(edge_lengths)) if len(edge_lengths) else None,
            }
            summaries.append(summary)
            print(f"[OK] {partition}/{trip}: nodes={n:,}, arcs={m:,}, route={r:,}, "
                  f"route_connected={summary['route_passed_directed']}")
    if not summaries:
        raise SystemExit('No valid trip folders found. Check --data-dir.')
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(out/'road_trip_summary.csv', index=False)
    pd.DataFrame(breaks).to_csv(out/'route_connectivity_breaks.csv', index=False)
    aggregate = {}
    for part, g in sdf.groupby('partition'):
        aggregate[part] = {
            'trip_count': int(len(g)),
            'total_nodes': int(g.node_count.sum()),
            'total_directed_arcs': int(g.directed_arc_count.sum()),
            'total_reference_route_arcs': int(g.reference_route_length.sum()),
            'trips_with_valid_arc_endpoints': int(g.arc_endpoints_valid.sum()),
            'trips_with_valid_route_arc_ids': int(g.route_arc_ids_valid.sum()),
            'routes_passed_connectivity': int(g.route_passed_directed.sum()),
            'routes_failed_connectivity': int((~g.route_passed_directed).sum()),
            'total_route_transitions': int(g.route_transitions.sum()),
            'total_connected_route_transitions': int(g.directed_connected_transitions.sum()),
            'duplicate_directed_arc_rows': int(g.duplicate_directed_arc_rows.sum()),
            'self_loop_arc_rows': int(g.self_loop_arc_rows.sum()),
            'duplicate_node_coordinate_rows': int(g.duplicate_node_coordinate_rows.sum()),
            'median_weak_component_count': float(g.weak_component_count.median()),
            'median_largest_weak_component_pct': float(g.largest_weak_component_pct.median()),
        }
    aggregate['combined'] = {
        'trip_count': int(len(sdf)),
        'total_nodes': int(sdf.node_count.sum()),
        'total_directed_arcs': int(sdf.directed_arc_count.sum()),
        'total_reference_route_arcs': int(sdf.reference_route_length.sum()),
        'routes_passed_connectivity': int(sdf.route_passed_directed.sum()),
        'routes_failed_connectivity': int((~sdf.route_passed_directed).sum()),
        'total_route_transitions': int(sdf.route_transitions.sum()),
        'total_connected_route_transitions': int(sdf.directed_connected_transitions.sum()),
        'route_transition_connectivity_pct': float(100*sdf.directed_connected_transitions.sum()/max(1,sdf.route_transitions.sum())),
        'duplicate_directed_arc_rows': int(sdf.duplicate_directed_arc_rows.sum()),
        'self_loop_arc_rows': int(sdf.self_loop_arc_rows.sum()),
        'duplicate_node_coordinate_rows': int(sdf.duplicate_node_coordinate_rows.sum()),
        'arc_length_median_m': float(np.median(all_edge_lengths)),
        'arc_length_p95_m': float(np.quantile(all_edge_lengths,.95)),
        'arc_length_p99_m': float(np.quantile(all_edge_lengths,.99)),
        'arc_length_max_m': float(np.max(all_edge_lengths)),
        'zero_length_arc_rows': int(np.sum(np.asarray(all_edge_lengths)==0)),
        'arc_rows_over_500m': int(np.sum(np.asarray(all_edge_lengths)>500)),
        'arc_rows_over_1000m': int(np.sum(np.asarray(all_edge_lengths)>1000)),
    }
    (out/'road_network_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    # Plot distributions (all arcs; use log scale because of long-tail counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_edge_lengths, bins=80)
    ax.set(title='Road-network arc length distribution', xlabel='Haversine edge length (metres)', ylabel='Arc rows')
    ax.set_yscale('log'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'arc_length_distribution.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_degrees, bins=np.arange(0, max(all_degrees)+2)-.5)
    ax.set(title='Road-network out-degree distribution', xlabel='Outgoing arcs per node', ylabel='Node count')
    ax.set_yscale('log'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'node_outdegree_distribution.png', dpi=160); plt.close(fig)
    print('\n=== PHASE 2.3 SUMMARY ===')
    print(json.dumps(aggregate, indent=2))
    print(f'\nSaved outputs to: {out.resolve()}')

if __name__ == '__main__':
    main()
