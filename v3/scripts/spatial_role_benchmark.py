"""Compare fixed model roles at frozen subject points without changing candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import spatial_camera_probe as P
import spatial_role_references as R

ROLE_MAP = {'player': 'skater', 'goalie': 'goalie', 'referee': 'referee'}
COUNT_NAMES = ('role_agreement', 'role_disagreement', 'ambiguous_point', 'unassociated_point',
               'unavailable_reference', 'unjudged_reference')


def compare_points(subjects: list[dict], detections: list[dict] | None, *, min_confidence: float = .25) -> dict:
    """Associate geometry before comparing roles; never resolve an ambiguous match by role."""
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError('invalid role comparison confidence')
    ids = set()
    for subject in subjects:
        x, y = subject['anchor_xy']
        if (not isinstance(subject['id'], str) or not subject['id'] or subject['id'] in ids
                or not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0
                or subject['status'] not in ('verified', 'uncertain')
                or subject['status'] == 'verified' and subject['role'] not in ROLE_MAP.values()
                or subject['status'] == 'uncertain' and subject['role'] != 'uncertain'):
            raise ValueError('invalid or duplicate role reference subject')
        ids.add(subject['id'])
    boxes = []
    for detection in detections or []:
        label = detection['class']
        confidence = float(detection['confidence'])
        box = [float(v) for v in detection['xyxy']]
        if (not isinstance(label, str) or not label or len(box) != 4
                or not all(math.isfinite(v) for v in box + [confidence]) or not 0 <= confidence <= 1
                or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('invalid role prediction box')
        if label in ROLE_MAP:
            boxes.append({'class': label, 'confidence': confidence, 'xyxy': box})
    boxes.sort(key=lambda b: (*b['xyxy'], b['class'], -b['confidence']))
    eligible = {i: box for i, box in enumerate(boxes) if box['confidence'] >= min_confidence}
    links = {}
    for subject in subjects:
        x, y = subject['anchor_xy']
        links[subject['id']] = [i for i, box in eligible.items()
                               if box['xyxy'][0] <= x < box['xyxy'][2] and box['xyxy'][1] <= y < box['xyxy'][3]]
    use_counts = Counter(i for indexes in links.values() for i in indexes)
    counts = {name: 0 for name in COUNT_NAMES}
    rows = []
    for subject in subjects:
        indexes = links[subject['id']]
        reasons = []
        if len(indexes) > 1:
            reasons.append('multiple_boxes')
        if any(use_counts[i] > 1 for i in indexes):
            reasons.append('shared_box')
        row = {'subject_id': subject['id'], 'reference_role': subject['role'],
               'anchor_xy': list(subject['anchor_xy']), 'ambiguity_reasons': reasons,
               'containing_boxes': [{'box_id': f'box:{i:03d}', **eligible[i]} for i in indexes]}
        if subject['status'] == 'uncertain':
            status = 'unjudged_reference'
        elif detections is None:
            status = 'unavailable_reference'
        elif reasons:
            status = 'ambiguous_point'
        elif not indexes:
            status = 'unassociated_point'
        else:
            row['predicted_role'] = ROLE_MAP[eligible[indexes[0]]['class']]
            status = 'role_agreement' if row['predicted_role'] == subject['role'] else 'role_disagreement'
        row['status'] = status
        counts[status] += 1
        rows.append(row)
    return {'status': 'point_role_comparison_only', 'min_confidence': min_confidence,
            'counts': counts, 'subjects': rows,
            'eligible_person_prediction_count': len(eligible),
            'unjudged_prediction_count': sum(use_counts[i] == 0 for i in eligible),
            'can_filter': False, 'can_reassign_camera': False}


def _key(row: dict) -> tuple:
    return row['game_id'], row['sample_number'], row['camera']


def benchmark_cached(references_path: Path, protocol_path: Path, frame_inventory_path: Path,
                     predictions_path: Path, association_path: Path) -> dict:
    """Validate cached center-frame predictions and keep development and evaluation results separate."""
    references = json.loads(references_path.read_text())
    R.validate_references(references, protocol_path, frame_inventory_path)
    frames = json.loads(frame_inventory_path.read_text())
    association = json.loads(association_path.read_text())
    predictions = json.loads(predictions_path.read_text())
    producer = predictions['producer']
    if (association['status'] != 'frozen_before_fixed_model_comparison'
            or association['references_sha256'] != P.sha256(references_path)
            or association['sampling_protocol_sha256'] != P.sha256(protocol_path)
            or association['model_weights_sha256'] != producer['weights_sha256']
            or association['confidence_thresholds'] != [.25, .15] or association['primary_threshold'] != .25
            or association['role_mapping'] != ROLE_MAP
            or any(producer[k] != value for k, value in association['inference'].items())
            or association['can_filter'] is not False or association['can_reassign_camera'] is not False
            or predictions['can_filter'] is not False or predictions['can_reassign_camera'] is not False
            or predictions['source_inventory_sha256'] != P.sha256(frame_inventory_path)
            or predictions['association_protocol_sha256'] != P.sha256(association_path)):
        raise ValueError('fixed role benchmark identity or settings changed')
    rows = {_key(row): row for row in predictions['records']}
    sources = {_key(row): row for row in frames['records']}
    if len(rows) != len(predictions['records']) or set(rows) != {_key(s) for s in references['samples']}:
        raise ValueError('fixed role predictions must retain every reference camera sample exactly once')
    results = {str(t): [] for t in association['confidence_thresholds']}
    for sample in references['samples']:
        row = rows[_key(sample)]
        source = sources[_key(sample)]
        if row['nominal_detection_s'] != sample['nominal_detection_s']:
            raise ValueError('role prediction nominal time changed')
        detections = None
        if row['status'] == 'observed':
            if sample['status'] != 'reviewed':
                raise ValueError('role predictions cannot infer missing reference frames')
            cache, prediction, roi = Path(row['center_cache_file']), Path(row['prediction_file']), Path(row['roi_file'])
            if (P.sha256(cache) != row['center_cache_sha256'] or P.sha256(prediction) != row['prediction_sha256']
                    or row['source_context_sha256'] != sample['frame_cache_sha256']
                    or row['center_cache_index'] != sample['center_cache_index']):
                raise ValueError('role prediction source artifact hash or index changed')
            with np.load(cache, allow_pickle=False) as data:
                identity = json.loads(str(data['identity'].item()))
                expected_identity = {**source['source_identity'], 'source_context_sha256': sample['frame_cache_sha256'],
                                     'center_cache_index': sample['center_cache_index']}
                image = cv2.imread(source['artifacts']['center']['file'])
                if (identity != expected_identity or data['frames'].shape != (1, *image.shape)
                        or not np.array_equal(data['frames'][0], image)
                        or not np.array_equal(data['times'], [sample['nominal_detection_s']])):
                    raise ValueError('role prediction center cache does not match the frozen reference frame')

            def forbidden(frame):
                raise ValueError('role benchmark requires cached predictions; it never runs inference')

            validated = P.analyze_cache(cache, roi, f"camera_{sample['camera'][3:]}", prediction,
                                        producer, forbidden, scope='source_selected')
            if len(validated['observations']) != 1:
                raise ValueError('role comparison requires exactly one frozen center-frame observation')
            observation = validated['observations'][0]
            detections = None if observation['evidence']['status'] == 'unavailable' else observation['evidence']['detections']
        elif row['status'] != 'unavailable':
            raise ValueError('role predictions require explicit observed or unavailable status')
        metadata = {k: sample[k] for k in ('game_id', 'sample_number', 'camera', 'partition', 'nominal_detection_s')}
        for threshold in association['confidence_thresholds']:
            results[str(threshold)].append({**metadata, **compare_points(sample['subjects'], detections,
                                                                        min_confidence=threshold)})
    summaries = {}
    for threshold, comparisons in results.items():
        def totals(items):
            return {name: sum(item['counts'][name] for item in items) for name in COUNT_NAMES}
        summaries[threshold] = {'all_selected_references': totals(comparisons),
                                'partitions': {partition: totals([item for item in comparisons if item['partition'] == partition])
                                               for partition in sorted({item['partition'] for item in comparisons})},
                                'games': {gid: totals([item for item in comparisons if item['game_id'] == gid])
                                          for gid in sorted({item['game_id'] for item in comparisons})}}
    return {'schema_version': 1, 'status': 'fixed_model_point_role_comparison_only',
            'input_identity': {str(path): P.sha256(path) for path in
                               (references_path, protocol_path, frame_inventory_path, predictions_path, association_path)},
            'code_sha256': P.sha256(Path(__file__)), 'producer': producer,
            'summaries': summaries, 'comparisons': results, 'limits': association['limits'],
            'can_filter': False, 'can_reassign_camera': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--frame-inventory', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--association-protocol', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = benchmark_cached(args.references, args.protocol, args.frame_inventory,
                                   args.predictions, args.association_protocol)
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, P.BadZipFile) as exc:
        print(f'role benchmark unavailable: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
