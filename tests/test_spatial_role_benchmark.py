import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v3/scripts'))
import spatial_role_benchmark as B


def subject(identifier='subject:001', x=15, role='goalie', status='verified'):
    return {'id': identifier, 'anchor_xy': [x, 15], 'role': role, 'status': status}


def box(label='player', confidence=.8, xyxy=None):
    return {'class': label, 'confidence': confidence, 'xyxy': xyxy or [10, 10, 20, 20]}


def test_unique_point_match_reports_goalie_mislabel_without_camera_authority():
    result = B.compare_points([subject()], [box()])
    assert result['counts']['role_disagreement'] == 1
    assert result['subjects'][0]['reference_role'] == 'goalie'
    assert result['subjects'][0]['predicted_role'] == 'skater'
    assert result['can_filter'] is result['can_reassign_camera'] is False


def test_player_class_maps_to_skater_reference_without_changing_boxes():
    boxes = [box()]
    before = copy.deepcopy(boxes)
    result = B.compare_points([subject(role='skater')], boxes)
    assert result['counts']['role_agreement'] == 1
    assert boxes == before


def test_box_association_never_uses_reference_roles_to_choose_overlapping_boxes():
    boxes = [box('goalie', .7), box('player', .9)]
    result = B.compare_points([subject()], boxes)
    assert result['counts']['ambiguous_point'] == 1
    assert result['counts']['role_agreement'] == 0
    assert result['counts']['role_disagreement'] == 0
    assert result['subjects'][0]['ambiguity_reasons'] == ['multiple_boxes']


def test_one_merged_box_cannot_match_two_people_even_when_roles_agree():
    result = B.compare_points([subject(x=14, role='skater'), subject('subject:002', x=16, role='skater')], [box()])
    assert result['counts']['ambiguous_point'] == 2
    assert result['counts']['role_agreement'] == 0
    assert all(s['ambiguity_reasons'] == ['shared_box'] for s in result['subjects'])


def test_uncertain_reference_remains_unjudged_and_prevents_shared_box_certification():
    result = B.compare_points([subject(), subject('subject:002', x=16, role='uncertain', status='uncertain')], [box('goalie')])
    assert result['counts']['unjudged_reference'] == 1
    assert result['counts']['ambiguous_point'] == 1
    assert result['counts']['role_agreement'] == 0


def test_unassociated_point_is_not_absence_or_whole_frame_detection_failure():
    result = B.compare_points([subject()], [box(xyxy=[30, 10, 40, 20])])
    assert result['counts']['unassociated_point'] == 1
    assert result['unjudged_prediction_count'] == 1
    assert 'false_positives' not in result
    assert 'absence_verified' not in result


def test_missing_predictions_remain_unavailable_not_an_empty_frame():
    missing = B.compare_points([subject()], None)
    empty = B.compare_points([subject()], [])
    assert missing['counts']['unavailable_reference'] == 1
    assert empty['counts']['unassociated_point'] == 1


def test_fixed_confidence_views_report_low_confidence_without_tuning():
    boxes = [box('goalie', .2)]
    primary = B.compare_points([subject()], boxes, min_confidence=.25)
    secondary = B.compare_points([subject()], boxes, min_confidence=.15)
    assert primary['counts']['unassociated_point'] == 1
    assert secondary['counts']['role_agreement'] == 1


def test_nonperson_boxes_do_not_supply_roles_and_unlabeled_people_are_unjudged():
    result = B.compare_points([subject()], [box('goal'), box('player', xyxy=[30, 10, 40, 20])])
    assert result['counts']['unassociated_point'] == 1
    assert result['unjudged_prediction_count'] == 1


def test_association_uses_half_open_containment():
    assert B.compare_points([subject(x=20)], [box()])['counts']['unassociated_point'] == 1
    assert B.compare_points([subject(x=10)], [box('goalie')])['counts']['role_agreement'] == 1


def test_reference_order_does_not_change_role_association_counts():
    subjects = [subject(), subject('subject:002', x=35, role='skater')]
    boxes = [box('goalie'), box('player', xyxy=[30, 10, 40, 20])]
    forward = B.compare_points(subjects, boxes)
    backward = B.compare_points(list(reversed(subjects)), list(reversed(boxes)))
    assert forward['counts'] == backward['counts']


@pytest.mark.parametrize('threshold', [-1, 2, float('nan')])
def test_invalid_confidence_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        B.compare_points([subject()], [], min_confidence=threshold)


@pytest.mark.parametrize('invalid', [box(confidence=float('inf')), box(xyxy=[10, 10, 10, 20])])
def test_malformed_model_boxes_are_not_scored_as_missing_roles(invalid):
    with pytest.raises(ValueError):
        B.compare_points([subject()], [invalid])


def test_duplicate_reference_ids_are_rejected():
    with pytest.raises(ValueError):
        B.compare_points([subject(), subject()], [box()])


def benchmark_inputs(tmp_path):
    game = tmp_path / 'game'
    game.mkdir()
    raw = game / 'raw.MP4'
    raw.write_bytes(b'synthetic source; never decode')
    image = np.zeros((100, 100, 3), np.uint8)
    center_image = tmp_path / 'center.png'
    context_image = tmp_path / 'context.png'
    assert cv2.imwrite(str(center_image), image) and cv2.imwrite(str(context_image), image)
    roi = game / 'rois.json'
    roi.write_text(json.dumps({'camera_1': {'net': [0, 0, 80, 80], 'slot': [0, 0, 80, 80]}}))
    identity = {'file': str(raw), 'size': raw.stat().st_size, 'mtime_ns': raw.stat().st_mtime_ns}
    context_cache = tmp_path / 'context.npz'
    np.savez(context_cache, frames=np.array([image]), times=np.array([10.]), identity=json.dumps(identity))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'status': 'frozen_before_new_frame_review', 'sample_offsets_s': [0],
        'width': 100, 'partitions': {'development': ['synthetic'], 'reserved_evaluation': ['other']},
        'can_filter': False, 'can_reassign_camera': False}))
    metadata = {'game_id': 'synthetic', 'sample_number': 1, 'camera': 'cam1',
                'partition': 'development', 'nominal_detection_s': 10.}
    artifacts = {name: {'file': str(path), 'sha256': B.P.sha256(path)} for name, path in
                 [('cache', context_cache), ('center', center_image), ('context', context_image)]}
    record = {**metadata, 'status': 'prepared', 'center_cache_index': 0, 'artifacts': artifacts,
              'source_identity': identity, 'protocol_sha256': B.P.sha256(protocol)}
    frame_inventory = tmp_path / 'frame_inventory.json'
    frame_inventory.write_text(json.dumps({'records': [record], 'protocol_sha256': B.P.sha256(protocol)}))
    reference = subject()
    reference.update(review_crop_xyxy=[10, 10, 20, 20], evidence_basis=['visible_equipment'],
                     evidence='Wide goalie pads are visible.', limitations=[])
    sample = {**metadata, 'status': 'reviewed', 'center_cache_index': 0, 'subjects': [reference],
              'scope': 'selected_visible_subjects_only', 'absence_verified': False, 'frame_size': [100, 100],
              'frame_cache_sha256': artifacts['cache']['sha256'], 'center_image_sha256': artifacts['center']['sha256'],
              'context_image_sha256': artifacts['context']['sha256']}
    references = tmp_path / 'references.json'
    references.write_text(json.dumps({'schema_version': 1, 'status': 'reviewed_role_references', 'samples': [sample],
        'protocol_sha256': B.P.sha256(protocol), 'inventory_sha256': B.P.sha256(frame_inventory),
        'can_filter': False, 'can_reassign_camera': False}))
    producer = {'weights_sha256': 'synthetic', 'prediction_confidence': .15, 'imgsz': 1280,
                'device': 'cpu', 'augment': False}
    association = tmp_path / 'association.json'
    association.write_text(json.dumps({'status': 'frozen_before_fixed_model_comparison',
        'references_sha256': B.P.sha256(references), 'sampling_protocol_sha256': B.P.sha256(protocol),
        'model_weights_sha256': 'synthetic', 'confidence_thresholds': [.25, .15], 'primary_threshold': .25,
        'role_mapping': B.ROLE_MAP, 'inference': {k: producer[k] for k in ('prediction_confidence', 'imgsz', 'device', 'augment')},
        'can_filter': False, 'can_reassign_camera': False, 'limits': ['Synthetic comparison only.']}))
    center_cache = tmp_path / 'center.npz'
    center_identity = {**identity, 'source_context_sha256': B.P.sha256(context_cache), 'center_cache_index': 0}
    np.savez(center_cache, frames=np.array([image]), times=np.array([10.]), identity=json.dumps(center_identity))
    prediction = tmp_path / 'prediction.json'
    B.P.analyze_cache(center_cache, roi, 'camera_1', prediction, producer, lambda frame: [box('goalie')], scope='source_selected')
    row = {**metadata, 'status': 'observed', 'prediction_file': str(prediction), 'prediction_sha256': B.P.sha256(prediction),
           'center_cache_file': str(center_cache), 'center_cache_sha256': B.P.sha256(center_cache), 'roi_file': str(roi),
           'source_context_sha256': B.P.sha256(context_cache), 'center_cache_index': 0}
    predictions = tmp_path / 'predictions.json'
    predictions.write_text(json.dumps({'records': [row], 'producer': producer,
        'source_inventory_sha256': B.P.sha256(frame_inventory), 'association_protocol_sha256': B.P.sha256(association),
        'can_filter': False, 'can_reassign_camera': False}))
    return references, protocol, frame_inventory, predictions, association


def test_cached_benchmark_is_read_only_and_validates_center_frame_identity(tmp_path):
    paths = benchmark_inputs(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = B.benchmark_cached(*paths)
    assert result['summaries']['0.25']['all_selected_references']['role_agreement'] == 1
    assert result['summaries']['0.25']['partitions']['development']['role_agreement'] == 1
    assert result['can_filter'] is result['can_reassign_camera'] is False
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize('change', ['producer', 'point_time', 'missing_sample', 'duplicate_sample', 'source_context'])
def test_changed_benchmark_inputs_are_rejected_without_inference(tmp_path, change):
    paths = benchmark_inputs(tmp_path)
    data = json.loads(paths[3].read_text())
    if change == 'producer':
        data['producer']['prediction_confidence'] = .3
    elif change == 'point_time':
        data['records'][0]['nominal_detection_s'] = 11.
    elif change == 'missing_sample':
        data['records'] = []
    elif change == 'duplicate_sample':
        data['records'] *= 2
    else:
        data['records'][0]['source_context_sha256'] = 'wrong'
    paths[3].write_text(json.dumps(data))
    with pytest.raises(ValueError):
        B.benchmark_cached(*paths)


def test_prediction_count_tampering_is_rejected_even_if_manifest_hash_is_rebound(tmp_path):
    paths = benchmark_inputs(tmp_path)
    inventory = json.loads(paths[3].read_text())
    prediction = Path(inventory['records'][0]['prediction_file'])
    data = json.loads(prediction.read_text())
    data['observations'][0]['evidence']['near_players'] = 500
    prediction.write_text(json.dumps(data))
    inventory['records'][0]['prediction_sha256'] = B.P.sha256(prediction)
    paths[3].write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='invalid cached spatial observation'):
        B.benchmark_cached(*paths)


def test_wrong_center_pixels_are_rejected_even_if_cache_hash_is_rebound(tmp_path):
    paths = benchmark_inputs(tmp_path)
    inventory = json.loads(paths[3].read_text())
    cache = Path(inventory['records'][0]['center_cache_file'])
    with np.load(cache, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays['frames'][:] = 1
    np.savez(cache, **arrays)
    inventory['records'][0]['center_cache_sha256'] = B.P.sha256(cache)
    paths[3].write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='center cache'):
        B.benchmark_cached(*paths)


def test_missing_predictions_do_not_silently_supply_absence_or_a_verified_role(tmp_path):
    paths = benchmark_inputs(tmp_path)
    inventory = json.loads(paths[3].read_text())
    inventory['records'][0] = {k: v for k, v in inventory['records'][0].items()
                              if k in ('game_id', 'sample_number', 'camera', 'nominal_detection_s')}
    inventory['records'][0].update(status='unavailable', error='Model output unavailable')
    paths[3].write_text(json.dumps(inventory))
    result = B.benchmark_cached(*paths)
    assert result['summaries']['0.25']['all_selected_references']['unavailable_reference'] == 1
    assert result['can_filter'] is False


def test_benchmark_cli_requires_existing_valid_predictions(tmp_path, capsys):
    paths = benchmark_inputs(tmp_path)
    args = [item for flag, path in zip(('--references', '--protocol', '--frame-inventory', '--predictions',
                                        '--association-protocol'), paths) for item in (flag, str(path))]
    assert B.main(args) == 0
    assert json.loads(capsys.readouterr().out)['can_reassign_camera'] is False
    paths[3].unlink()
    assert B.main(args) == 2
    assert 'benchmark unavailable' in capsys.readouterr().err
