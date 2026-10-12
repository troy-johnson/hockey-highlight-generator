import copy
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v3/scripts'))
import spatial_role_references as R


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path):
    raw = tmp_path / 'raw.MP4'
    raw.write_bytes(b'synthetic source; never decode')
    image = np.zeros((72, 128, 3), np.uint8)
    cache = tmp_path / 'context.npz'
    identity = {'file': str(raw), 'size': raw.stat().st_size, 'mtime_ns': raw.stat().st_mtime_ns}
    np.savez(cache, frames=np.stack([image] * 9), times=np.arange(9) * .5 + 8., identity=json.dumps(identity))
    center = tmp_path / 'center.png'
    context = tmp_path / 'context.png'
    assert cv2.imwrite(str(center), image) and cv2.imwrite(str(context), image)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'schema_version': 1, 'status': 'frozen_before_new_frame_review',
        'sample_offsets_s': [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2], 'width': 128,
        'partitions': {'development': ['synthetic'], 'reserved_evaluation': ['other']},
        'can_filter': False, 'can_reassign_camera': False}))
    artifacts = {name: {'file': str(path), 'sha256': digest(path)} for name, path in
                 [('cache', cache), ('center', center), ('context', context)]}
    record = {'game_id': 'synthetic', 'sample_number': 1, 'camera': 'cam1', 'partition': 'development',
        'nominal_detection_s': 10., 'status': 'prepared', 'center_cache_index': 4,
        'source_identity': identity, 'artifacts': artifacts, 'protocol_sha256': digest(protocol)}
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps({'protocol_sha256': digest(protocol), 'records': [record]}))
    sample = {'game_id': 'synthetic', 'sample_number': 1, 'camera': 'cam1', 'partition': 'development',
        'nominal_detection_s': 10., 'status': 'reviewed', 'center_cache_index': 4,
        'frame_cache_sha256': artifacts['cache']['sha256'], 'center_image_sha256': artifacts['center']['sha256'],
        'context_image_sha256': artifacts['context']['sha256'], 'frame_size': [128, 72],
        'scope': 'selected_visible_subjects_only', 'absence_verified': False,
        'subjects': [{'id': 'subject:001', 'anchor_xy': [50, 40], 'review_crop_xyxy': [40, 20, 60, 60],
                     'role': 'goalie', 'status': 'verified', 'evidence_basis': ['visible_equipment'],
                     'evidence': 'Wide leg pads and a catcher glove are visible.', 'limitations': []}]}
    references = {'schema_version': 1, 'status': 'reviewed_role_references', 'protocol_sha256': digest(protocol),
        'inventory_sha256': digest(inventory), 'can_filter': False, 'can_reassign_camera': False, 'samples': [sample]}
    return references, protocol, inventory, raw


def test_role_references_bind_raw_cache_and_image_without_changing_inputs(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    before = copy.deepcopy(data)
    result = R.validate_references(data, protocol, inventory)
    assert result['verified_roles'] == {'goalie': 1}
    assert result['sample_count'] == 1
    assert result['can_filter'] is result['can_reassign_camera'] is False
    assert data == before


@pytest.mark.parametrize('basis', [['model_label'], ['net_position_only'], []])
def test_model_label_or_net_position_alone_is_not_verified_ground_truth(tmp_path, basis):
    data, protocol, inventory, raw = inputs(tmp_path)
    data['samples'][0]['subjects'][0]['evidence_basis'] = basis
    with pytest.raises(ValueError, match='evidence'):
        R.validate_references(data, protocol, inventory)


def test_uncertain_person_is_retained_without_a_confirmed_role(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    subject = data['samples'][0]['subjects'][0]
    subject.update(role='uncertain', status='uncertain', evidence_basis=['occluded'],
                   evidence='The net obscures the equipment.', limitations=['Equipment cannot be identified.'])
    result = R.validate_references(data, protocol, inventory)
    assert result['uncertain_subjects'] == 1
    assert result['verified_roles'] == {}


@pytest.mark.parametrize('field,value', [('role', 'goal'), ('status', 'verified'),
                                        ('limitations', [])])
def test_uncertain_reference_requires_explicit_uncertainty(tmp_path, field, value):
    data, protocol, inventory, raw = inputs(tmp_path)
    subject = data['samples'][0]['subjects'][0]
    subject.update(role='uncertain', status='uncertain', evidence_basis=['occluded'],
                   evidence='The equipment is obscured.', limitations=['Equipment cannot be identified.'])
    subject[field] = value
    with pytest.raises(ValueError):
        R.validate_references(data, protocol, inventory)


@pytest.mark.parametrize('field,value', [('anchor_xy', [float('nan'), 40]), ('anchor_xy', [128, 40]),
    ('review_crop_xyxy', [40, 20, 40, 60]), ('review_crop_xyxy', [40, 20, 130, 60]),
    ('review_crop_xyxy', [60, 20, 70, 60])])
def test_invalid_or_unbound_subject_geometry_is_rejected(tmp_path, field, value):
    data, protocol, inventory, raw = inputs(tmp_path)
    data['samples'][0]['subjects'][0][field] = value
    with pytest.raises(ValueError):
        R.validate_references(data, protocol, inventory)


def test_missing_or_duplicate_sample_cannot_hide_difficult_reference(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    duplicated = copy.deepcopy(data)
    duplicated['samples'] *= 2
    for invalid in (duplicated, {**data, 'samples': []}):
        with pytest.raises(ValueError, match='sample'):
            R.validate_references(invalid, protocol, inventory)


@pytest.mark.parametrize('field,value', [('partition', 'reserved_evaluation'), ('absence_verified', True),
                                       ('nominal_detection_s', 11.), ('center_cache_index', 3)])
def test_role_labels_cannot_change_partition_time_absence_or_anchor_frame(tmp_path, field, value):
    data, protocol, inventory, raw = inputs(tmp_path)
    data['samples'][0][field] = value
    with pytest.raises(ValueError):
        R.validate_references(data, protocol, inventory)


def test_changed_raw_file_refuses_reference_validation(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    raw.write_bytes(b'different raw source')
    with pytest.raises(ValueError, match='source'):
        R.validate_references(data, protocol, inventory)


def test_changed_image_bytes_refuse_reference_validation(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    (tmp_path / 'center.png').write_bytes(b'changed image')
    with pytest.raises(ValueError, match='hash'):
        R.validate_references(data, protocol, inventory)


def test_rebound_center_hash_cannot_substitute_a_different_cache_frame(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    center = tmp_path / 'center.png'
    assert cv2.imwrite(str(center), np.ones((72, 128, 3), np.uint8))
    saved = json.loads(inventory.read_text())
    saved['records'][0]['artifacts']['center']['sha256'] = digest(center)
    inventory.write_text(json.dumps(saved))
    data['inventory_sha256'] = digest(inventory)
    data['samples'][0]['center_image_sha256'] = digest(center)
    with pytest.raises(ValueError, match='center image'):
        R.validate_references(data, protocol, inventory)


def test_unavailable_planned_sample_is_retained_without_fabricated_roles(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    saved = json.loads(inventory.read_text())
    saved['records'][0].update(status='unavailable', error='Missing camera coverage')
    inventory.write_text(json.dumps(saved))
    data['inventory_sha256'] = digest(inventory)
    data['samples'][0].update(status='unavailable', subjects=[])
    result = R.validate_references(data, protocol, inventory)
    assert result['unavailable_samples'] == 1
    assert result['verified_roles'] == {}


def test_reference_metadata_never_authorizes_camera_changes(tmp_path):
    data, protocol, inventory, raw = inputs(tmp_path)
    data['can_reassign_camera'] = True
    with pytest.raises(ValueError):
        R.validate_references(data, protocol, inventory)


@pytest.mark.parametrize('role,basis', [('goalie', ['visible_skating']),
                                     ('referee', ['visible_equipment']), ('skater', ['occluded'])])
def test_verified_roles_require_role_specific_visible_evidence(tmp_path, role, basis):
    data, protocol, inventory, raw = inputs(tmp_path)
    data['samples'][0]['subjects'][0].update(role=role, evidence_basis=basis)
    with pytest.raises(ValueError, match='evidence'):
        R.validate_references(data, protocol, inventory)


def test_role_reference_cli_is_read_only_and_reports_scope_limits(tmp_path, capsys):
    data, protocol, inventory, raw = inputs(tmp_path)
    references = tmp_path / 'references.json'
    references.write_text(json.dumps(data))
    before = {p: p.read_bytes() for p in (references, protocol, inventory, raw)}
    assert R.main(['--references', str(references), '--protocol', str(protocol), '--inventory', str(inventory)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['verified_roles'] == {'goalie': 1}
    assert any('not visual label correctness' in limit for limit in result['limits'])
    assert all(p.read_bytes() == value for p, value in before.items())


def test_role_reference_cli_refuses_missing_artifacts(tmp_path, capsys):
    data, protocol, inventory, raw = inputs(tmp_path)
    references = tmp_path / 'references.json'
    references.write_text(json.dumps(data))
    (tmp_path / 'context.png').unlink()
    assert R.main(['--references', str(references), '--protocol', str(protocol), '--inventory', str(inventory)]) == 2
    assert 'validation failed' in capsys.readouterr().err


def test_frozen_role_dataset_is_separate_from_outcome_keys_and_preserves_partitions():
    root = Path(__file__).resolve().parent / 'role_references'
    suite = json.loads((root / 'suite.json').read_text())
    references = json.loads((root / suite['references']).read_text())
    protocol = json.loads((root / suite['protocol']).read_text())
    assert suite['dataset_kind'] == references['dataset_kind'] == 'independent_visual_role_references'
    assert references['protocol_sha256'] == digest(root / suite['protocol'])
    assert references['can_filter'] is references['can_reassign_camera'] is False
    assert set(protocol['partitions']['development']).isdisjoint(protocol['partitions']['reserved_evaluation'])
    assert protocol['rank_fractions'] == [.375, .625]
    assert len(references['samples']) == 20
    assert len({(s['game_id'], s['sample_number'], s['camera']) for s in references['samples']}) == 20
    assert all(s['game_id'] in protocol['partitions'][s['partition']] for s in references['samples'])
    assert all(s['absence_verified'] is False for s in references['samples'])
    assert all(subject['role'] not in ('goal', 'non_goal') for s in references['samples'] for subject in s['subjects'])
    assert any(subject['status'] == 'uncertain' for s in references['samples'] for subject in s['subjects'])
