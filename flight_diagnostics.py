"""Retrospective diagnostics; never changes controls or success criteria."""
import numpy as np


def takeoff_progress(trajectory):
    """Detect a near-full-throttle, low-speed plateau over the final 3 s.

    Diagnostic heuristic: speed below 4 m/s throughout, span <=0.5 m/s,
    and throttle >=0.95 throughout. A short window is explicitly insufficient.
    This does not establish that every possible controller would fail.
    """
    if not trajectory:
        return dict(low_speed_plateau=False, sufficient_history=False)
    end = trajectory[-1]['t']
    window = [row for row in trajectory if row['t'] >= end - 3.0 - 1e-8]
    duration = end - window[0]['t']
    speeds = [row['Vx'] for row in window]
    enough = duration >= 3.0 - 1e-8
    plateau = (enough and min(speeds) >= 0 and max(speeds) < 4
               and max(speeds)-min(speeds) <= .5
               and min(row['throttle'] for row in window) >= .95)
    result = dict(low_speed_plateau=bool(plateau), sufficient_history=bool(enough),
                  window_seconds=float(duration), min_speed_m_s=float(min(speeds)),
                  max_speed_m_s=float(max(speeds)),
                  speed_change_m_s=float(speeds[-1]-speeds[0]))
    durations = np.diff([row['t'] for row in window])
    def weighted(key, nested=None):
        values = [row[nested][key] if nested else row[key] for row in window]
        if not durations.size or durations.sum() <= 0:
            return float(values[-1])
        return float(np.average(values[1:], weights=durations))
    if all('force_budget' in row for row in window):
        result['mean_force_budget'] = {
            key: weighted(key, 'force_budget') for key in window[-1]['force_budget']}
    if all('T_factor' in row for row in window):
        result['mean_T_factor'] = weighted('T_factor')
    if all('prop_clearance_m' in row for row in window):
        result['mean_prop_clearance_m'] = weighted('prop_clearance_m')
        result['min_prop_clearance_m'] = float(min(row['prop_clearance_m'] for row in window))
    if all('prop_bottom_clearance_m' in row for row in window):
        result['min_prop_bottom_clearance_m'] = float(
            min(row['prop_bottom_clearance_m'] for row in window))
    return result
