"""Validate separate visual role references; never change candidates or model roles."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import BadZipFile

import cv2
import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(sample: dict) -> tuple:
    return sample['game_id'], sample['sample_number'], sample['camera']


def validate_references(references: dict, protocol_path: Path, inventory_path: Path) -> dict:
    """Check reference provenance and uncertainty, not whether the visual role is correct."""
    protocol = json.loads(protocol_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    if (references['schema_version'] != 1 or references['status'] != 'reviewed_role_references'
            or references['can_filter'] is not False or references['can_reassign_camera'] is not False
            or protocol['status'] != 'frozen_before_new_frame_review'
            or protocol['can_filter'] is not False or protocol['can_reassign_camera'] is not False
            or references['protocol_sha256'] != _sha(protocol_path)
            or references['inventory_sha256'] != _sha(inventory_path)
            or inventory['protocol_sha256'] != _sha(protocol_path)):
        raise ValueError('invalid role reference provenance or authorization')
    records = {_key(row): row for row in inventory['records']}
    samples = references['samples']
    if (len(records) != len(inventory['records']) or len(samples) != len(records)
            or len({_key(s) for s in samples}) != len(samples)
            or {_key(s) for s in samples} != set(records)):
        raise ValueError('role references must retain every planned camera sample exactly once')
    roles = Counter()
    partitions = Counter()
    uncertain = unavailable = 0
    for sample in samples:
        record = records[_key(sample)]
        partition = sample['partition']
        if (partition != record['partition'] or partition not in protocol['partitions']
                or sample['game_id'] not in protocol['partitions'][partition]
                or sample['nominal_detection_s'] != record['nominal_detection_s']
                or not math.isfinite(sample['nominal_detection_s'])
                or sample['absence_verified'] is not False
                or sample['scope'] != 'selected_visible_subjects_only'
                or record['protocol_sha256'] != _sha(protocol_path)):
            raise ValueError('invalid role sample partition, time, or scope')
        partitions[partition] += 1
        if record['status'] == 'unavailable':
            if sample['status'] != 'unavailable' or sample['subjects']:
                raise ValueError('unavailable role sample cannot have inferred subjects')
            unavailable += 1
            continue
        if record['status'] != 'prepared' or sample['status'] != 'reviewed':
            raise ValueError('role sample must have explicit review status')
        artifacts = record['artifacts']
        for name, field in (('cache', 'frame_cache_sha256'), ('center', 'center_image_sha256'),
                            ('context', 'context_image_sha256')):
            path = Path(artifacts[name]['file'])
            if _sha(path) != artifacts[name]['sha256'] or sample[field] != artifacts[name]['sha256']:
                raise ValueError('role reference artifact hash changed')
        identity = record['source_identity']
        stat = Path(identity['file']).stat()
        if stat.st_size != identity['size'] or stat.st_mtime_ns != identity['mtime_ns']:
            raise ValueError('role reference raw source identity changed')
        with np.load(artifacts['cache']['file'], allow_pickle=False) as data:
            frames, times = data['frames'], data['times']
            if json.loads(str(data['identity'].item())) != identity:
                raise ValueError('role cache source identity mismatch')
        index = sample['center_cache_index']
        expected_times = sample['nominal_detection_s'] + np.array(protocol['sample_offsets_s'])
        if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
                or times.ndim != 1 or len(times) != len(frames) or not np.isfinite(times).all()
                or len(times) != len(expected_times) or not np.allclose(times, expected_times, rtol=0, atol=1e-9)
                or not isinstance(index, int) or index != record['center_cache_index']
                or not 0 <= index < len(times) or abs(times[index] - sample['nominal_detection_s']) > 1e-9):
            raise ValueError('role reference cache frame or nominal time changed')
        height, width = frames.shape[1:3]
        image = cv2.imread(artifacts['center']['file'])
        if (sample['frame_size'] != [width, height] or width != protocol['width']
                or image is None or not np.array_equal(image, frames[index])):
            raise ValueError('role reference center image does not match the bound cache frame')
        ids = set()
        for subject in sample['subjects']:
            if not isinstance(subject['id'], str) or not subject['id'] or subject['id'] in ids:
                raise ValueError('duplicate or invalid role subject ID')
            ids.add(subject['id'])
            x, y = subject['anchor_xy']
            x0, y0, x1, y1 = subject['review_crop_xyxy']
            if (not all(math.isfinite(v) for v in (x, y, x0, y0, x1, y1))
                    or not 0 <= x0 <= x < x1 <= width or not 0 <= y0 <= y < y1 <= height):
                raise ValueError('invalid role reference subject point or crop')
            basis = subject['evidence_basis']
            allowed = {'visible_equipment', 'visible_skating', 'referee_stripes', 'occluded', 'insufficient_detail'}
            if (not basis or not set(basis).issubset(allowed)
                    or not isinstance(subject['evidence'], str) or not subject['evidence'].strip()
                    or not isinstance(subject['limitations'], list)
                    or not all(isinstance(v, str) and v.strip() for v in subject['limitations'])):
                raise ValueError('invalid role reference visual evidence')
            if subject['status'] == 'uncertain':
                if subject['role'] != 'uncertain' or not subject['limitations']:
                    raise ValueError('uncertain role requires an explicit uncertain label and limits')
                uncertain += 1
            elif subject['status'] == 'verified':
                required = {'goalie': {'visible_equipment'},
                            'skater': {'visible_equipment', 'visible_skating'},
                            'referee': {'referee_stripes'}}
                if subject['role'] not in required or not set(basis).intersection(required[subject['role']]):
                    raise ValueError('verified role requires visible role evidence')
                roles[subject['role']] += 1
            else:
                raise ValueError('unknown role reference review status')
    return {'status': 'reference_provenance_validated', 'sample_count': len(samples),
            'partition_samples': dict(partitions), 'verified_roles': dict(roles),
            'uncertain_subjects': uncertain, 'unavailable_samples': unavailable,
            'can_filter': False, 'can_reassign_camera': False,
            'limits': ['Validation checks provenance and schema, not visual label correctness.',
                       'Selected subjects do not enumerate all people or establish absence.',
                       'Reference points and crops are not full-body boxes or tracks.',
                       'These references do not establish role accuracy or camera-filter safety.']}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_references(json.loads(args.references.read_text()), args.protocol, args.inventory)
    except (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, BadZipFile) as exc:
        print(f'role reference validation failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
