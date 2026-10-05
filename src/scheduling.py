"""One-day hourly scheduling from the project's direct max15 P90 forecast.

This is a bounded, deterministic recommendation search, not a plant controller.
Supplied task loads are constant additive assumptions in the source power unit.
Simulation removes the original task load before adding the recommended load;
it does not refit the predictor or establish measured, causal power savings.
"""
from math import isfinite
from numbers import Real


SEARCH_BUDGET = 50000
_TOL = 1e-9


def _integer(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f'{label}: {low}~{high} 사이의 정수가 필요합니다.')
    return value


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{label}: 유한한 숫자가 필요합니다.')
    value = float(value)
    if not isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{label}: 유한한 {"양수" if positive else "0 이상 숫자"}가 필요합니다.')
    return value


def _forecast_rows(forecast):
    if not isinstance(forecast, list) or len(forecast) != 24:
        raise ValueError('예측은 0~23시를 모두 포함하는 24개 행이어야 합니다.')
    rows = {}
    for row in forecast:
        if not isinstance(row, dict):
            raise ValueError('각 예측 행은 객체여야 합니다.')
        hour = _integer(row.get('hour'), '예측 hour', 0, 23)
        if hour in rows:
            raise ValueError(f'예측 {hour}시가 중복되었습니다.')
        rows[hour] = {'hour': hour}
        for key in ('pred', 'pred_max15', 'pred_hi_max15', 'alarm_threshold'):
            rows[hour][key] = _number(row.get(key), f'{hour}시 {key}', key == 'alarm_threshold')
    return [rows[hour] for hour in range(24)]


def _tasks(tasks):
    if not isinstance(tasks, list):
        raise ValueError('작업은 목록이어야 합니다.')
    # An explicit bound keeps the browser request within this one-day prototype.
    if len(tasks) > 96:
        raise ValueError('하루 작업은 최대 96개까지 입력할 수 있습니다.')
    normalized = []
    ids = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('각 작업은 객체여야 합니다.')
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not task_id.strip() or task_id in ids:
            raise ValueError('작업 ID는 비어 있지 않은 고유 문자열이어야 합니다.')
        ids.add(task_id)
        value = {'task_id': task_id, 'name': task.get('name', task_id)}
        if not isinstance(value['name'], str):
            raise ValueError(f'{task_id}: 작업 이름은 문자열이어야 합니다.')
        for key, low, high in (
            ('original_start', 0, 23), ('duration', 1, 24),
            ('earliest_start', 0, 23), ('latest_end', 1, 24),
        ):
            value[key] = _integer(task.get(key), f'{task_id} {key}', low, high)
        if value['earliest_start'] + value['duration'] > value['latest_end']:
            raise ValueError(f'{task_id}: 허용 시간대에 작업을 완료할 수 없습니다.')
        value['load'] = None if task.get('load') is None else _number(task['load'], f'{task_id} load')
        value['movable'] = task.get('movable', True)
        if not isinstance(value['movable'], bool):
            raise ValueError(f'{task_id}: movable은 참/거짓이어야 합니다.')
        value['resource'] = task.get('resource', '')
        if not isinstance(value['resource'], str):
            raise ValueError(f'{task_id}: resource는 문자열이어야 합니다.')
        dependencies = task.get('depends_on', [])
        if (not isinstance(dependencies, list)
                or any(not isinstance(dep, str) for dep in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError(f'{task_id}: 선행 작업 ID는 중복 없는 문자열 목록이어야 합니다.')
        value['depends_on'] = list(dependencies)
        normalized.append(value)
    by_id = {task['task_id']: task for task in normalized}
    for task in normalized:
        for dep in task['depends_on']:
            if dep not in by_id:
                raise ValueError(f'{task["task_id"]}: 선행 작업 {dep}가 없습니다.')
    # A topological order makes precedence enforceable during partial search.
    pending = set(ids)
    order = []
    while pending:
        eligible = [by_id[key] for key in pending
                    if all(dep not in pending for dep in by_id[key]['depends_on'])]
        if not eligible:
            raise ValueError('선행 작업 관계에 순환이 있습니다.')
        task = min(eligible, key=lambda item: (
            1 if item['movable'] else 0,
            item['latest_end'] - item['earliest_start'] - item['duration'],
            -item['duration'], -(item['load'] or 0), item['task_id'],
        ))
        order.append(task)
        pending.remove(task['task_id'])
    return normalized, order, by_id


def _overlap(start_a, duration_a, start_b, duration_b):
    return start_a < start_b + duration_b and start_b < start_a + duration_a


def _valid_baseline(tasks, by_id, max_parallel):
    counts = [0] * 24
    for task in tasks:
        start, duration = task['original_start'], task['duration']
        if start < task['earliest_start'] or start + duration > task['latest_end']:
            raise ValueError(f'{task["task_id"]}: 기존 일정이 허용 시간대를 벗어납니다.')
        for dep_id in task['depends_on']:
            dep = by_id[dep_id]
            if dep['original_start'] + dep['duration'] > start:
                raise ValueError(f'{task["task_id"]}: 기존 일정이 선행 작업 순서를 위반합니다.')
        for hour in range(start, start + duration):
            counts[hour] += 1
    if any(count > max_parallel for count in counts):
        raise ValueError('기존 일정이 최대 동시 작업 수를 초과합니다.')
    for i, task in enumerate(tasks):
        if not task['resource']:
            continue
        for other in tasks[i + 1:]:
            if task['resource'] == other['resource'] and _overlap(
                task['original_start'], task['duration'], other['original_start'], other['duration'],
            ):
                raise ValueError(f'기존 일정에서 자원 {task["resource"]}의 작업이 겹칩니다.')
    return counts


def _loads(tasks, starts):
    result = [0.] * 24
    for task in tasks:
        for hour in range(starts[task['task_id']], starts[task['task_id']] + task['duration']):
            result[hour] += task['load'] or 0.
    return result


def _risk(counts, p90, thresholds):
    exposure = sum(count * max(value / threshold - 1., 0.)
                   for count, value, threshold in zip(counts, p90, thresholds))
    alarm_hours = sum(count for count, value, threshold in zip(counts, p90, thresholds)
                      if value >= threshold)
    return float(exposure), int(alarm_hours)


def _exceedance(values, thresholds):
    return float(sum(max(value - threshold, 0.) for value, threshold in zip(values, thresholds)))


def schedule_day(forecast, tasks, max_parallel=2):
    """Recommend all jobs within their declared windows and precedence constraints.

    Invalid original schedules are rejected, never silently repaired. The original
    valid schedule is an incumbent: an interrupted search cannot worsen it.
    Missing or inconsistent task loads select risk-avoidance mode, which makes no
    claim about the recommended schedule's actual or predicted power reduction.
    """
    rows = _forecast_rows(forecast)
    max_parallel = _integer(max_parallel, 'max_parallel', 1, 96)
    tasks, order, by_id = _tasks(tasks)
    baseline_counts = _valid_baseline(tasks, by_id, max_parallel)
    original = {task['task_id']: task['original_start'] for task in tasks}
    p90 = [row['pred_hi_max15'] for row in rows]
    thresholds = [row['alarm_threshold'] for row in rows]
    known_loads = bool(tasks) and all(task['load'] is not None for task in tasks)
    baseline_loads = _loads(tasks, original) if known_loads else [None] * 24
    fits_total = known_loads and all(
        load <= min(row['pred_max15'], row['pred_hi_max15']) + _TOL
        for row, load in zip(rows, baseline_loads)
    )
    mode = 'load_simulation' if fits_total else 'risk_avoidance'
    warnings = [
        '1시간 단위의 하루 추천입니다. 작업 수와 소요 시간, 고정 작업, 허용 시간대와 선행 관계를 유지합니다.',
        '입력하지 않은 생산량·인력·공정·설비 안전 조건은 검증하지 않습니다. 현장 확인 후 적용해야 합니다.',
        '학습 Q95는 프로젝트 상대 경보 기준입니다. 계약전력 초과나 요금 절감으로 해석하지 않습니다.',
    ]
    if mode == 'load_simulation':
        warnings.append('저감량은 사용자가 입력한 일정 부하를 가산·이동한 시뮬레이션입니다. 실측 효과나 인과 효과가 아닙니다.')
        warnings.append('P90 곡선에 부하 이동을 가산한 가정 시나리오입니다. 변경 일정으로 재학습한 P90이나 보정된 신뢰상한이 아닙니다.')
        warnings.append('부하 단위는 예측 데이터의 원래 전력 단위와 같아야 합니다. 전력·에너지 단위는 별도 확인이 필요합니다.')
    elif not tasks:
        warnings.append('작업이 없어 기존 예측만 반환합니다.')
    elif not known_loads:
        warnings.append('설비별 부하가 모두 입력되지 않아 경보 시간의 작업 노출만 줄입니다. 저감 전력 수치는 산출하지 않습니다.')
    else:
        warnings.append('기존 작업 부하 합계가 예측 전체 부하를 초과하여 부하 시뮬레이션을 사용하지 않습니다. 부하 입력을 확인하세요.')

    background = [max(value - load, 0.) for value, load in zip(p90, baseline_loads)] if fits_total else p90
    counts = [0] * 24
    allocated = [0.] * 24
    assigned = {}
    best_starts = dict(original)
    baseline_risk = _risk(baseline_counts, p90, thresholds)

    def objective(power, occupancy, movement):
        if mode == 'load_simulation':
            return (round(max(power), 9), round(_exceedance(power, thresholds), 9), movement)
        risk = _risk(occupancy, p90, thresholds)
        return (round(risk[0], 9), risk[1], movement)

    original_objective = objective(p90, baseline_counts, 0)
    best_objective = original_objective
    nodes = 0
    exhausted = False

    def allowed(task, start):
        if any(counts[hour] >= max_parallel for hour in range(start, start + task['duration'])):
            return False
        if any(assigned[dep] + by_id[dep]['duration'] > start for dep in task['depends_on']):
            return False
        if task['resource']:
            for key, other_start in assigned.items():
                other = by_id[key]
                if other['resource'] == task['resource'] and _overlap(
                    start, task['duration'], other_start, other['duration'],
                ):
                    return False
        # A future fixed successor must still be reachable.
        for successor in order:
            if task['task_id'] in successor['depends_on']:
                latest = successor['original_start'] if not successor['movable'] else successor['latest_end'] - successor['duration']
                if start + task['duration'] > latest:
                    return False
        return True

    def search(position, movement):
        nonlocal nodes, exhausted, best_objective, best_starts
        if nodes >= SEARCH_BUDGET:
            exhausted = True
            return
        nodes += 1
        partial_power = [base + load for base, load in zip(background, allocated)]
        lower = objective(partial_power, counts, movement)
        # Every remaining contribution and movement cost is nonnegative.
        if lower > best_objective:
            return
        if position == len(order):
            if lower < best_objective:
                best_objective = lower
                best_starts = dict(assigned)
            return
        task = order[position]
        candidates = ([task['original_start']] if not task['movable'] else
                      range(task['earliest_start'], task['latest_end'] - task['duration'] + 1))
        candidates = [start for start in candidates if allowed(task, start)]

        def priority(start):
            proposed_counts = list(counts)
            proposed_power = list(partial_power)
            for hour in range(start, start + task['duration']):
                proposed_counts[hour] += 1
                if fits_total:
                    proposed_power[hour] += task['load']
            score = objective(proposed_power, proposed_counts, movement + abs(start - task['original_start']))
            return score, start

        for start in sorted(candidates, key=priority):
            assigned[task['task_id']] = start
            for hour in range(start, start + task['duration']):
                counts[hour] += 1
                if fits_total:
                    allocated[hour] += task['load']
            search(position + 1, movement + abs(start - task['original_start']))
            for hour in range(start, start + task['duration']):
                counts[hour] -= 1
                if fits_total:
                    allocated[hour] -= task['load']
            del assigned[task['task_id']]
            if exhausted:
                break

    search(0, 0)
    recommended_counts = [0] * 24
    output_tasks = []
    for task in tasks:
        start = best_starts[task['task_id']]
        for hour in range(start, start + task['duration']):
            recommended_counts[hour] += 1
        output_tasks.append(dict(task, recommended_start=start, changed=start != task['original_start']))
    recommended_loads = _loads(tasks, best_starts) if known_loads else [None] * 24
    after_p90 = [base + load for base, load in zip(background, recommended_loads)] if fits_total else None
    after_max15 = [max(row['pred_max15'] - old, 0.) + new for row, old, new in
                   zip(rows, baseline_loads, recommended_loads)] if fits_total else None
    average_fits = fits_total and all(load <= row['pred'] + _TOL for row, load in zip(rows, baseline_loads))
    after_average = [max(row['pred'] - old, 0.) + new for row, old, new in
                     zip(rows, baseline_loads, recommended_loads)] if average_fits else None
    if fits_total and not average_fits:
        warnings.append('작업 부하가 시간 평균 예측을 초과하는 시간이 있어 시간 평균 전력은 시뮬레이션하지 않습니다. 15분 최대만 비교합니다.')
    if exhausted:
        warnings.append('탐색 예산에 도달했습니다. 확인한 후보 중 기존 일정 이상인 추천을 반환하며 전역 최적해를 보장하지 않습니다.')
    if best_objective == original_objective:
        warnings.append('현재 조건에서 기존 일정보다 나은 후보를 찾지 못해 기존 일정을 유지합니다.')
    after_risk = _risk(recommended_counts, p90, thresholds)
    modeled_energy = float(sum(task['load'] * task['duration'] for task in tasks)) if fits_total else None
    hourly = []
    for hour, row in enumerate(rows):
        hourly.append(dict(
            row, baseline_alarm=p90[hour] >= thresholds[hour],
            after_pred=after_average[hour] if after_average is not None else None,
            after_pred_max15=after_max15[hour] if after_max15 is not None else None,
            after_pred_hi_max15=after_p90[hour] if after_p90 is not None else None,
            after_alarm=after_p90[hour] >= thresholds[hour] if after_p90 is not None else None,
            baseline_task_load=baseline_loads[hour], recommended_task_load=recommended_loads[hour],
            baseline_task_count=baseline_counts[hour], recommended_task_count=recommended_counts[hour],
        ))
    return {
        'mode': mode, 'tasks': output_tasks, 'hourly': hourly,
        'metrics': {
            'before_peak': float(max(p90)), 'after_peak': float(max(after_p90)) if fits_total else None,
            'peak_reduction': float(max(p90) - max(after_p90)) if fits_total else None,
            'before_exceedance': _exceedance(p90, thresholds),
            'after_exceedance': _exceedance(after_p90, thresholds) if fits_total else None,
            'before_alarm_hours': sum(value >= threshold for value, threshold in zip(p90, thresholds)),
            'after_alarm_hours': sum(value >= threshold for value, threshold in zip(after_p90, thresholds)) if fits_total else None,
            'before_risk_exposure': baseline_risk[0], 'after_risk_exposure': after_risk[0],
            'before_alarm_task_hours': baseline_risk[1], 'after_alarm_task_hours': after_risk[1],
            'changed_tasks': sum(task['changed'] for task in output_tasks),
            'total_shift_hours': sum(abs(best_starts[task['task_id']] - task['original_start']) for task in tasks),
            'workload_hours': sum(task['duration'] for task in tasks),
            'modeled_task_energy_before': modeled_energy, 'modeled_task_energy_after': modeled_energy,
        },
        'warnings': warnings,
        'search': {'algorithm': 'bounded_branch_and_bound', 'nodes': nodes, 'budget': SEARCH_BUDGET,
                   'complete': not exhausted, 'improved': best_objective < original_objective,
                   'objective_before': list(original_objective), 'objective_after': list(best_objective)},
    }
