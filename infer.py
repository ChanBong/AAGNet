"""
Standalone inference script for AAGNet machining feature recognition.

Takes a STEP file and outputs structured JSON with per-face feature
type labels, instance IDs, bottom faces, and confidence scores.

Usage:
    python infer.py path/to/part.step
    python infer.py path/to/part.step --output results.json
    python infer.py path/to/part.step --device cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Compound, TopoDS_CompSolid
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer

from dataset.AAGExtractor import AAGExtractor
from dataset.topologyCheker import TopologyChecker
from models.inst_segmentors import AAGNetSegmentor
from utils.data_utils import (
    load_one_graph,
    load_json_or_pkl,
    load_statistics,
)

# 25 classes: 24 machining feature types + stock
FEAT_NAMES = [
    "chamfer",
    "through_hole",
    "triangular_passage",
    "rectangular_passage",
    "6sides_passage",
    "triangular_through_slot",
    "rectangular_through_slot",
    "circular_through_slot",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "Oring",
    "blind_hole",
    "triangular_pocket",
    "rectangular_pocket",
    "6sides_pocket",
    "circular_end_pocket",
    "rectangular_blind_slot",
    "v_circular_end_blind_slot",
    "h_circular_end_blind_slot",
    "triangular_blind_step",
    "circular_blind_step",
    "rectangular_blind_step",
    "round",
    "stock",
]

STOCK_CLASS = 24  # index of "stock" in FEAT_NAMES

# Paths relative to repo root
REPO_ROOT = Path(__file__).resolve().parent
WEIGHT_PATH = REPO_ROOT / "weights" / "weight_on_MFInstseg.pth"
ATTR_SCHEMA_PATH = REPO_ROOT / "feature_lists" / "all.json"
ATTR_STAT_PATH = REPO_ROOT / "weights" / "attr_stat.json"


def load_body_from_step(step_file):
    """Load shape from STEP file."""
    from OCC.Core.STEPControl import STEPControl_Reader
    import pathlib
    assert pathlib.Path(step_file).suffix.lower() in ['.step', '.stp']
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_file))
    reader.TransferRoots()
    return reader.OneShape()


def extract_solids(shape):
    """
    Extract individual TopoDS_Solid bodies from a shape.
    If the shape is already a Solid, returns it as-is.
    If it's a Compound/CompSolid, iterates child solids.
    Returns list of (solid, face_offset) tuples where face_offset
    is the index of the first face of this solid in the compound-level
    TopologyExplorer ordering.
    """
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods

    if isinstance(shape, TopoDS_Solid):
        return [(shape, 0)]

    # Compound or CompSolid: extract child solids and compute face offsets
    # First, enumerate ALL faces in compound-level order to build offset map
    compound_topo = TopologyExplorer(shape)
    compound_faces = list(compound_topo.faces())
    compound_face_hashes = {f.__hash__(): i for i, f in enumerate(compound_faces)}

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solid = topods.Solid(explorer.Current())
        # Find face offset: the compound-level index of this solid's first face
        solid_topo = TopologyExplorer(solid)
        solid_faces = list(solid_topo.faces())
        if solid_faces:
            first_hash = solid_faces[0].__hash__()
            face_offset = compound_face_hashes.get(first_hash, -1)
        else:
            face_offset = -1
        solids.append((solid, face_offset))
        explorer.Next()

    return solids


def load_model(device: str) -> AAGNetSegmentor:
    """Load AAGNet model with pretrained weights."""
    model = AAGNetSegmentor(
        arch="AAGNetGraphEncoder",
        num_classes=len(FEAT_NAMES),
        edge_attr_dim=12,
        node_attr_dim=10,
        edge_attr_emb=64,
        node_attr_emb=64,
        edge_grid_dim=0,
        node_grid_dim=7,
        edge_grid_emb=0,
        node_grid_emb=64,
        num_layers=3,
        delta=2,
        mlp_ratio=2,
        drop=0.0,
        drop_path=0.0,
        head_hidden_dim=64,
        conv_on_edge=False,
    )
    model_param = torch.load(str(WEIGHT_PATH), map_location="cpu")
    model.load_state_dict(model_param)
    model = model.to(device)
    model.eval()
    return model


def _infer_single_solid(
    solid,
    model: AAGNetSegmentor,
    device: str,
    attribute_schema: dict,
    stat: dict,
    step_path_str: str,
    face_offset: int = 0,
    inst_thres: float = 0.5,
    bottom_thres: float = 0.5,
) -> dict:
    """
    Run AAGNet on one TopoDS_Solid.  Returns features with face IDs
    offset by `face_offset` so they map to compound-level indexing.
    """
    eps = 1e-6

    topo = TopologyExplorer(solid)
    num_faces = topo.number_of_faces()

    checker = TopologyChecker()
    if not checker(solid):
        return {"error": "Topology check failed", "num_faces": num_faces}

    # --- Extract AAG (works on a TopoDS_Solid directly via AAGExtractor) ---
    try:
        extractor = AAGExtractor(step_path_str, attribute_schema)
        # Override the loader — we already have the solid
        extractor.body = solid
        from dataset.AAGExtractor import scale_solid_to_unit_box
        from OCC.Core.TopoDS import TopoDS_Solid as _Solid
        if extractor.scale_body:
            extractor.body = scale_solid_to_unit_box(extractor.body)
        # Validate
        assert extractor.topchecker(extractor.body), \
            "Topology check failed after scaling"
        # Build face adjacency graph
        from occwl.solid import Solid as OccwlSolid
        from occwl.graph import face_adjacency as _fa
        occwl_solid = OccwlSolid(extractor.body)
        # Remove assembly transforms so face surface queries work
        occwl_solid.set_transform_to_identity()
        graph_nx = _fa(occwl_solid)

        # Extract face attributes
        graph_face_attr = []
        graph_face_grid = []
        len_of_face_attr = len(extractor.attribute_schema["face_attributes"]) + \
            2 if "FaceCentroidAttribute" in extractor.attribute_schema["face_attributes"] else 0
        for face_idx in graph_nx.nodes:
            face = graph_nx.nodes[face_idx]["face"]
            face_attr = extractor.extract_attributes_from_face(face.topods_shape())
            assert len_of_face_attr == len(face_attr)
            graph_face_attr.append(face_attr)
            if extractor.use_uv and extractor.num_srf_u and extractor.num_srf_v:
                uv_grid = extractor.extract_face_point_grid(face)
                assert uv_grid.shape[0] == 7
                graph_face_grid.append(uv_grid.tolist())

        graph_edge_attr = []
        graph_edge_grid = []
        for edge_idx in graph_nx.edges:
            edge = graph_nx.edges[edge_idx]["edge"]
            if not edge.has_curve():
                continue
            edge_topo = edge.topods_shape()
            edge_attr = extractor.extract_attributes_from_edge(edge_topo)
            assert len(extractor.attribute_schema["edge_attributes"]) == len(edge_attr)
            graph_edge_attr.append(edge_attr)
            if extractor.use_uv and extractor.num_crv_u:
                u_grid = extractor.extract_edge_point_grid(edge_topo)
                assert u_grid.shape[0] == 12
                graph_edge_grid.append(u_grid.tolist())

        edges = list(graph_nx.edges)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        aag = {
            'graph': {'edges': (src, dst), 'num_nodes': len(graph_nx.nodes)},
            'graph_face_attr': graph_face_attr,
            'graph_face_grid': graph_face_grid,
            'graph_edge_attr': graph_edge_attr,
            'graph_edge_grid': graph_edge_grid,
        }
    except Exception as e:
        return {"error": f"AAG extraction failed: {e}", "num_faces": num_faces}

    # --- Build DGL graph ---
    try:
        sample = load_one_graph(step_path_str, aag)
        graph = sample["graph"]

        if graph.num_nodes() < 2 or graph.num_edges() == 0:
            return {"error": f"Degenerate shape: {graph.num_nodes()} nodes, {graph.num_edges()} edges",
                    "num_faces": num_faces}

        if graph.ndata["x"].numel() > 0:
            graph.ndata["x"] = ((graph.ndata["x"] - stat["mean_face_attr"]) / stat["std_face_attr"]).float()
        if graph.edata["x"].numel() > 0:
            graph.edata["x"] = ((graph.edata["x"] - stat["mean_edge_attr"]) / stat["std_edge_attr"]).float()
        graph = graph.to(device)
    except Exception as e:
        return {"error": f"Graph construction failed: {e}", "num_faces": num_faces}

    # --- Model forward pass ---
    try:
        with torch.no_grad():
            seg_out, inst_out, bottom_out = model(graph)
    except Exception as e:
        return {"error": f"Inference failed: {e}", "num_faces": num_faces}

    # --- Post-process ---
    face_logits = seg_out.cpu().numpy()
    face_probs = torch.softmax(seg_out, dim=1).cpu().numpy()

    inst_matrix = inst_out[0].sigmoid()
    adj = (inst_matrix > inst_thres).cpu().numpy().astype("int32")

    bottom_probs = bottom_out.sigmoid().cpu().numpy().flatten()
    bottom_mask = bottom_probs > bottom_thres

    # Group faces into instances
    proposals = []
    used = np.zeros(adj.shape[0], dtype=bool)
    for row_idx in range(adj.shape[0]):
        if used[row_idx]:
            continue
        if np.sum(adj[row_idx]) <= eps:
            continue
        group = set()
        for col_idx in range(adj.shape[1]):
            if used[col_idx]:
                continue
            if adj[row_idx][col_idx]:
                group.add(col_idx)
                used[col_idx] = True
        if group:
            proposals.append(sorted(group))

    # Build features with offset face IDs
    features = []
    feature_face_set = set()

    for local_face_ids in proposals:
        sum_logit = np.sum([face_logits[f] for f in local_face_ids], axis=0)
        predicted_class = int(np.argmax(sum_logit))
        if predicted_class == STOCK_CLASS:
            continue

        per_face_confs = [float(face_probs[f][predicted_class]) for f in local_face_ids]
        avg_confidence = float(np.mean(per_face_confs))
        global_face_ids = [f + face_offset for f in local_face_ids]
        global_bottom_ids = [f + face_offset for f in local_face_ids if bottom_mask[f]]

        features.append({
            "feature_type": FEAT_NAMES[predicted_class],
            "face_ids": global_face_ids,
            "bottom_face_ids": global_bottom_ids,
            "confidence": round(avg_confidence, 4),
        })
        feature_face_set.update(global_face_ids)

    stock_faces = sorted(
        set(range(face_offset, face_offset + num_faces)) - feature_face_set
    )

    return {
        "features": features,
        "stock_faces": stock_faces,
        "num_faces": num_faces,
        "aag_nodes": aag["graph"]["num_nodes"],
        "aag_edges": len(aag["graph"]["edges"][0]),
    }


def infer_step_file(
    step_file: str,
    model: AAGNetSegmentor,
    device: str,
    attribute_schema: dict,
    stat: dict,
    inst_thres: float = 0.5,
    bottom_thres: float = 0.5,
) -> dict:
    """
    Run AAGNet inference on a single STEP file.
    Handles both TopoDS_Solid and TopoDS_Compound shapes.
    """
    step_path = Path(step_file)
    t0 = time.time()

    # --- Load shape ---
    try:
        shape = load_body_from_step(str(step_path))
    except Exception as e:
        return {"part_name": step_path.name, "error": f"Failed to load: {e}", "total_faces": 0}

    topo = TopologyExplorer(shape)
    total_faces = topo.number_of_faces()

    # --- Extract solids (handles Compound/CompSolid) ---
    solids = extract_solids(shape)
    if not solids:
        return {
            "part_name": step_path.name,
            "error": "No solids found in shape",
            "total_faces": total_faces,
        }

    is_compound = len(solids) > 1

    # --- Process each solid ---
    all_features = []
    all_stock_faces = []
    all_aag_nodes = 0
    all_aag_edges = 0
    solid_errors = []
    instance_id = 1

    for solid_idx, (solid, face_offset) in enumerate(solids):
        result = _infer_single_solid(
            solid, model, device, attribute_schema, stat,
            str(step_path), face_offset,
            inst_thres, bottom_thres,
        )

        if "error" in result:
            solid_errors.append(f"solid[{solid_idx}]: {result['error']}")
            continue

        for feat in result["features"]:
            feat["instance_id"] = instance_id
            all_features.append(feat)
            instance_id += 1

        all_stock_faces.extend(result["stock_faces"])
        all_aag_nodes += result.get("aag_nodes", 0)
        all_aag_edges += result.get("aag_edges", 0)

    t_end = time.time()

    output = {
        "part_name": step_path.name,
        "features": all_features,
        "stock_faces": sorted(all_stock_faces),
        "total_faces": total_faces,
        "aag_nodes": all_aag_nodes,
        "aag_edges": all_aag_edges,
        "timings": {"total_s": round(t_end - t0, 4)},
    }

    if is_compound:
        output["num_solids"] = len(solids)

    if solid_errors:
        output["solid_errors"] = solid_errors

    return output


def main():
    parser = argparse.ArgumentParser(
        description="AAGNet inference: STEP file -> machining feature segmentation"
    )
    parser.add_argument("step_file", type=str, help="Path to a STEP file (.step/.stp)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--inst-threshold",
        type=float,
        default=0.5,
        help="Instance segmentation threshold (default: 0.5)",
    )
    parser.add_argument(
        "--bottom-threshold",
        type=float,
        default=0.5,
        help="Bottom face classification threshold (default: 0.5)",
    )
    args = parser.parse_args()

    step_path = Path(args.step_file)
    if not step_path.exists():
        print(f"Error: {step_path} does not exist", file=sys.stderr)
        sys.exit(1)
    if step_path.suffix.lower() not in (".step", ".stp"):
        print(f"Error: {step_path} is not a STEP file", file=sys.stderr)
        sys.exit(1)

    # Load model and config
    attribute_schema = load_json_or_pkl(str(ATTR_SCHEMA_PATH))
    stat = load_statistics(str(ATTR_STAT_PATH))
    model = load_model(args.device)

    # Run inference
    result = infer_step_file(
        str(step_path),
        model,
        args.device,
        attribute_schema,
        stat,
        inst_thres=args.inst_threshold,
        bottom_thres=args.bottom_threshold,
    )

    # Output
    json_str = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Results written to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
