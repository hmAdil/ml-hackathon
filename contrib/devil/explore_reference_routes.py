from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_table(path, dtype=float):
    a=np.loadtxt(path, dtype=dtype, ndmin=2)
    return a

def haversine(lon1, lat1, lon2, lat2):
    r=6371000.0
    p1=np.radians(lat1); p2=np.radians(lat2)
    dp=p2-p1; dl=np.radians(lon2-lon1)
    x=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(x,0,1)))

def analyze_trip(route_path):
    tid=route_path.stem
    trip_dir=route_path.parent
    nodes=load_table(trip_dir/f"{tid}.nodes")
    arcs=load_table(trip_dir/f"{tid}.arcs", dtype=int)
    route=np.loadtxt(route_path,dtype=int,ndmin=1).reshape(-1)
    valid=(route>=0)&(route<len(arcs))
    selected=arcs[route[valid]]
    endpoints_valid=(arcs[:,0]>=0)&(arcs[:,0]<len(nodes))&(arcs[:,1]>=0)&(arcs[:,1]<len(nodes))
    selected_endpoints_valid=(selected[:,0]>=0)&(selected[:,0]<len(nodes))&(selected[:,1]>=0)&(selected[:,1]<len(nodes))
    selected=selected[selected_endpoints_valid]
    if len(selected):
        lengths=haversine(nodes[selected[:,0],0],nodes[selected[:,0],1],
                          nodes[selected[:,1],0],nodes[selected[:,1],1])
        self_loops=int(np.sum(selected[:,0]==selected[:,1]))
    else:
        lengths=np.array([])
        self_loops=0
    transitions=(selected[:-1,1]==selected[1:,0]) if len(selected)>1 else np.array([],dtype=bool)
    return {
        "split":trip_dir.parent.name, "trip_id":tid,
        "node_count":len(nodes), "arc_count":len(arcs),
        "route_arc_count":len(route), "valid_route_arc_ids":int(valid.sum()),
        "invalid_route_arc_ids":int((~valid).sum()),
        "invalid_arc_endpoint_rows":int((~endpoints_valid).sum()),
        "unique_route_arc_ids":int(len(np.unique(route[valid]))),
        "repeated_route_arc_occurrences":int(valid.sum()-len(np.unique(route[valid]))),
        "route_length_km":float(lengths.sum()/1000),
        "median_selected_arc_m":float(np.median(lengths)) if len(lengths) else 0,
        "max_selected_arc_m":float(np.max(lengths)) if len(lengths) else 0,
        "self_loop_occurrences":self_loops,
        "connected_transitions":int(transitions.sum()),
        "broken_transitions":int((~transitions).sum()),
        "transition_count":int(len(transitions)),
        "all_transitions_connected":bool(transitions.all()) if len(transitions) else True
    }

def main():
    ap=argparse.ArgumentParser(description="Explore PathSnap reference routes")
    ap.add_argument("--data-dir",required=True,help="Path to extracted Student folder")
    ap.add_argument("--output-dir",default="outputs/phase2_4")
    args=ap.parse_args()
    root=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    route_files=sorted(root.glob("*/ */*.route"))
    # Explicitly find train and student_test trip folders
    route_files=sorted(root.glob("*/**/*.route"))
    rows=[]
    for rp in route_files:
        tid=rp.stem
        if (rp.parent/f"{tid}.nodes").exists() and (rp.parent/f"{tid}.arcs").exists():
            rows.append(analyze_trip(rp))
    if not rows:
        raise SystemExit(f"No complete .route/.nodes/.arcs trip folders found under {root}")
    df=pd.DataFrame(rows).sort_values(["split","trip_id"])
    df.to_csv(out/"reference_route_trip_summary.csv",index=False)
    breaks=df[df.broken_transitions>0]
    breaks.to_csv(out/"reference_route_connectivity_issues.csv",index=False)
    summary={}
    for split,g in df.groupby("split"):
        summary[split]={
            "trips":int(len(g)),
            "total_route_arcs":int(g.route_arc_count.sum()),
            "median_route_arcs":float(g.route_arc_count.median()),
            "median_route_length_km":float(g.route_length_km.median()),
            "route_length_km_p95":float(g.route_length_km.quantile(.95)),
            "min_route_length_km":float(g.route_length_km.min()),
            "max_route_length_km":float(g.route_length_km.max()),
            "trips_with_broken_transitions":int((g.broken_transitions>0).sum()),
            "total_broken_transitions":int(g.broken_transitions.sum()),
            "trips_with_repeated_arc_ids":int((g.repeated_route_arc_occurrences>0).sum()),
            "total_repeated_arc_occurrences":int(g.repeated_route_arc_occurrences.sum())
        }
    summary["overall"]={
        "trips":int(len(df)),
        "total_route_arcs":int(df.route_arc_count.sum()),
        "valid_route_arc_ids":int(df.valid_route_arc_ids.sum()),
        "invalid_route_arc_ids":int(df.invalid_route_arc_ids.sum()),
        "connected_transitions":int(df.connected_transitions.sum()),
        "broken_transitions":int(df.broken_transitions.sum()),
        "route_transition_connectivity_pct":float(100*df.connected_transitions.sum()/max(1,df.transition_count.sum())),
        "total_route_length_km_sum":float(df.route_length_km.sum())
    }
    (out/"reference_route_summary.json").write_text(json.dumps(summary,indent=2))
    for col,title,xlabel,filename in [
        ("route_length_km","Reference route length distribution","Route length (km)","route_length_distribution.png"),
        ("route_arc_count","Reference route arc-count distribution","Number of arcs","route_arc_count_distribution.png")]:
        plt.figure(figsize=(8,5)); plt.hist(df[col],bins=20); plt.title(title); plt.xlabel(xlabel); plt.ylabel("Trips"); plt.tight_layout(); plt.savefig(out/filename,dpi=150); plt.close()
    print(df.groupby("split").agg(
        trips=("trip_id","count"), total_route_arcs=("route_arc_count","sum"),
        median_route_arcs=("route_arc_count","median"), median_route_km=("route_length_km","median"),
        trips_with_breaks=("broken_transitions",lambda s:int((s>0).sum())),
        broken_transitions=("broken_transitions","sum"),
        trips_with_repeated_arcs=("repeated_route_arc_occurrences",lambda s:int((s>0).sum()))
    ).to_string())
    print("\nOVERALL:",json.dumps(summary["overall"],indent=2))
    print(f"\nSaved outputs to: {out.resolve()}")

if __name__=="__main__": main()
