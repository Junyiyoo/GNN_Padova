"""Build shared-boundary and centroid-distance graphs in the supplied node order."""
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

def read_order(path, index_column):
    # Edge endpoints are integer positions in this sorted node table.
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r[index_column]))
    ids = [r["SEZ21_ID"].strip() for r in rows]
    if not ids or "" in ids or len(set(ids)) != len(ids):
        raise ValueError('Section identifiers must be nonempty and unique.')
    if [int(r[index_column]) for r in rows] != list(range(len(ids))):
        raise ValueError(f'Node indices must be contiguous from zero. {index_column}')
    return ids

def read_zones(path, ids, layer_name=None):
    'Read polygons, project them to EPSG:32632 and align them to section IDs.'
    from osgeo import ogr, osr

    dataset = ogr.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayerByName(layer_name) if layer_name else dataset.GetLayer(0)
    if layer is None or layer.GetSpatialRef() is None:
        raise ValueError('Missing census layer or coordinate reference system.')
    if layer.FindFieldIndex("SEZ21_ID", 1) < 0:
        raise ValueError('The census layer requires SEZ21_ID.')
    source = layer.GetSpatialRef().Clone()
    target = osr.SpatialReference()
    target.ImportFromEPSG(32632)
    for crs in (source, target):
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = None if source.IsSame(target) else osr.CoordinateTransformation(source, target)
    zones = {}
    wanted = set(ids)
    for feature in layer:
        key = feature.GetFieldAsString("SEZ21_ID").strip()
        if key not in wanted:
            continue
        geom = feature.GetGeometryRef()
        if key in zones or geom is None:
            raise ValueError(f'Duplicate section ID or missing geometry. {key}')
        # Keep an independent geometry after GDAL releases the feature.
        shape = geom.Clone()
        if transform is not None and shape.Transform(transform) != 0:
            raise ValueError(f'Coordinate transformation failed. {key}')
        if shape.GetGeometryName() not in {"POLYGON", "MULTIPOLYGON"} or shape.IsEmpty() or not shape.IsValid():
            raise ValueError(f'Expected a valid census polygon. {key}')
        zones[key] = shape
    missing = wanted - zones.keys()
    if missing:
        raise ValueError(f'Some requested census polygons are missing. {len(missing)}, {sorted(missing)[:5]}')
    return [zones[key] for key in ids]

def boundary_pairs(zones, min_length_m=0.0):
    'Keep pairs sharing a boundary segment longer than the specified threshold.'
    if not np.isfinite(min_length_m) or min_length_m < 0:
        raise ValueError('Shared-boundary length must be finite and nonnegative.')
    boxes = np.array([g.GetEnvelope() for g in zones])  # xmin, xmax, ymin, ymax
    boundaries = [g.Boundary() for g in zones]
    pairs = []
    for i, (xmin, xmax, ymin, ymax) in enumerate(boxes):

        # Check only overlapping bounding boxes before intersecting the boundaries.
        rest = boxes[i + 1:]
        hits = (rest[:, 0] <= xmax) & (rest[:, 1] >= xmin) & (rest[:, 2] <= ymax) & (rest[:, 3] >= ymin)
        for j in np.flatnonzero(hits) + i + 1:
            intersection = boundaries[i].Intersection(boundaries[j])
            if intersection is None:
                raise ValueError(f'Boundary intersection failed. {i}, {j}')
            # A single shared corner has zero length.
            if intersection.GetDimension() == 1 and intersection.Length() > min_length_m:
                pairs.append((i, j))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

def undirected_pairs(edge_index, n):
    'Deduplicate directed records into undirected pairs, excluding self-loops.'
    e = np.asarray(edge_index)
    if e.ndim != 2 or e.shape[0] != 2 or e.dtype.kind not in "iu":
        raise ValueError('Expected a 2 by E integer edge array.')
    if np.any((e < 0) | (e >= n)):
        raise ValueError('An edge endpoint falls outside the node table.')
    pairs = np.sort(e.T, axis=1)
    return np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)

def restrict_pairs(pairs, original_ids, selected_ids):
    'Restrict the original topology and remap endpoints to the selected order.'
    position = {key: i for i, key in enumerate(original_ids)}
    remap = np.full(len(original_ids), -1, dtype=np.int64)
    remap[[position[key] for key in selected_ids]] = np.arange(len(selected_ids))
    mapped = remap[pairs]
    return np.unique(np.sort(mapped[(mapped >= 0).all(axis=1)], axis=1), axis=0)

def euclidean_knn(xy, k=8):
    'Use stable kNN ranking, symmetric union and the median positive edge length.'
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or not 0 < k < len(xy):
        raise ValueError('Expected finite projected coordinates and k smaller than the node count.')
    distances = cdist(xy, xy)
    np.fill_diagonal(distances, np.inf)
    # Break distance ties in node order, then keep the union of directed choices.
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :k]
    edges = np.stack([np.repeat(np.arange(len(xy)), k), nearest.ravel()])
    pairs = undirected_pairs(edges, len(xy))
    d = distances[pairs[:, 0], pairs[:, 1]]
    if not (d > 0).any():
        raise ValueError('The selected edges contain no positive distances.')
    # Use the median positive edge length as the Gaussian distance scale.
    sigma = float(np.median(d[d > 0]))
    return pairs, d, np.exp(-(d / sigma) ** 2).astype(np.float32), sigma

def bidirectional(pairs):
    """Store each undirected edge in both message-passing directions."""
    return np.concatenate([pairs.T, pairs[:, ::-1].T], axis=1)
