"""Reduced spatial dynamics with commanded bank and yaw rate.

Not a six-DOF rigid body: pitch is commanded directly, bank follows a
first-order servo in air, and yaw combines coordinated-turn and rudder.
On water, under-wing floats add buoyancy and a roll restoring moment.
"""
import math
import numpy as np
from dynamics import hull_force, float_contacts


def integrate(ac, hull, hd, atmosphere, surface_at, state, *, dt, t,
              pitch, throttle, bank_command, rudder_command,
              extra_mass=0.0, thrust_factor=1.0):
    """Integrate [x,y,z,Vx,Vy,Vz,bank,heading] without mutating callers.

    World axes: north/east/up. Forces are world-frame newtons.
    Thrust loss and added water mass are supplied by the damage adapter.
    """
    state = np.asarray(state, dtype=float)
    controls = [dt, t, pitch, throttle, bank_command, rudder_command,
                extra_mass, thrust_factor]
    if (state.shape != (8,) or not np.isfinite(state).all()
            or not np.isfinite(controls).all() or dt <= 0
            or extra_mass < 0 or not 0 <= thrust_factor <= 1):
        raise ValueError('invalid spatial state or integration parameters')
    x, y, z, vx, vy, vz, bank, heading = state
    n_sub = max(1, math.ceil(dt / 0.01))
    h = dt / n_sub
    mass = ac.mass.total + extra_mass
    force_sums = np.zeros(3)  # thrust, aero, water; forward components
    initial_vx = vx
    for i in range(n_sub):
        now = t + i * h
        surface = surface_at(x, y, now)
        wind = atmosphere.wind(now, z)
        velocity = np.array([vx, vy, vz])
        air = velocity - wind
        speed = float(np.linalg.norm(air))
        right = np.array([-math.sin(heading), math.cos(heading), 0.0])
        eta_left = surface_at(x - ac.geom.float_y * right[0],
                              y - ac.geom.float_y * right[1], now)
        eta_right = surface_at(x + ac.geom.float_y * right[0],
                               y + ac.geom.float_y * right[1], now)
        contact = hull_force(z, math.hypot(vx, vy), vz, surface, hull)
        left, right_float = float_contacts(ac, z, bank, vx, vz, eta_left, eta_right)
        waterborne = (contact.N + left.N + right_float.N) / (mass * 9.80665)
        hull_borne = contact.N > 0.2 * mass * 9.80665
        # A float grazing a crest does not rob the ailerons of authority;
        # only the main hull riding on water desensitizes the servo.
        servo = 0.05 if hull_borne else 1.0
        bank += (bank_command - bank) * (1 - math.exp(-h / 0.5)) * servo
        if waterborne > 0.0:
            # Buoyancy righting is a spring toward the wave slope across the
            # floats, not a free angle impulse: bounded by construction.
            slope = math.atan2(eta_left - eta_right, 2.0 * ac.geom.float_y)
            support = min(1.0, waterborne)
            bank += (slope - bank) * (1 - math.exp(-h / 0.3)) * support
        bank = float(np.clip(bank, -math.radians(45), math.radians(45)))
        # Coordinated turn approximation, only when airborne.
        turn = 9.80665 * math.tan(bank) / max(speed, 3.0) if not hull_borne else 0.0
        heading += (turn + rudder_command * math.radians(20)) * h
        heading = math.atan2(math.sin(heading), math.cos(heading))
        right = np.array([-math.sin(heading), math.cos(heading), 0.0])
        forward = np.array([math.cos(pitch) * math.cos(heading),
                            math.cos(pitch) * math.sin(heading), math.sin(pitch)])
        thrust_axis = np.array([
            math.cos(pitch + ac.aero.alpha_T) * math.cos(heading),
            math.cos(pitch + ac.aero.alpha_T) * math.sin(heading),
            math.sin(pitch + ac.aero.alpha_T)])
        up = np.cross(forward, right)
        # Positive bank tilts the lift toward +body-y.
        wing_right = right * math.cos(bank) - up * math.sin(bank)
        tangent = air / speed if speed > 1e-8 else forward
        lift_dir = np.cross(tangent, wing_right)
        norm = float(np.linalg.norm(lift_dir))
        lift_dir = lift_dir / norm if norm > 1e-8 else np.zeros(3)
        alpha = math.atan2(-float(air @ up), float(air @ forward))
        beta = math.asin(float(np.clip(air @ right / max(speed, 1e-8), -1, 1)))
        density = atmosphere.density(z)
        q = 0.5 * density * speed ** 2
        cl = ac.CL(alpha)
        lift = q * ac.geom.S * cl
        drag = q * ac.geom.S * (ac.CD(cl, height_m=z - surface) + 0.3 * beta ** 2)
        thrust = ac.prop.thrust(speed, throttle, rho=density) * thrust_factor
        force = thrust * thrust_axis + lift * lift_dir - drag * tangent
        thrust_x = thrust * thrust_axis[0]
        aero_x = force[0] - thrust_x
        water_x = 0.0
        horizontal = math.hypot(vx, vy)
        water_n = contact.N + left.N + right_float.N
        if water_n > 0 and horizontal > 1e-8:
            resistance = (hd.resistance(horizontal) if contact.N > 0 else 0.0) \
                + hull.mu_s * contact.N + 0.05 * (left.N + right_float.N)
            water_x = -resistance * velocity[0] / horizontal
            force[:2] -= resistance * velocity[:2] / horizontal
        force_sums += [thrust_x, aero_x, water_x]
        force[2] += water_n - (mass * 9.80665)
        velocity += force / mass * h
        vx, vy, vz = map(float, velocity)
        x += vx * h
        y += vy * h
        z += vz * h
    result = np.array([x, y, z, vx, vy, vz, bank, heading])
    if not np.isfinite(result).all():
        raise FloatingPointError('spatial integration produced a nonfinite state')
    mean_forces = force_sums / n_sub
    budget = dict(thrust_x_N=float(mean_forces[0]), aerodynamic_x_N=float(mean_forces[1]),
                  water_x_N=float(mean_forces[2]), net_x_N=float(mean_forces.sum()),
                  mean_ax_m_s2=float((vx-initial_vx)/dt), total_mass_kg=float(mass))
    return result, dict(N_water=contact.N, T=thrust, L=lift, D=drag, force_budget=budget)


def advance(env, x, z, vx, vz, pitch, throttle, sea, t):
    """Compatibility adapter for the RL environment."""
    eta = env._eta(x, t)
    veta = (env._eta(x, t + env.cfg.dt) - eta) / env.cfg.dt
    state, forces = integrate(
        env.ac, env.hull, env.hd, env._atmosphere,
        lambda x, y, t: env._eta(x, t, y),
        [x, env._y, z, vx, env._Vy, vz, env._bank, env._heading],
        dt=env.cfg.dt, t=t, pitch=pitch, throttle=throttle,
        bank_command=env._bank_command, rudder_command=env._rudder_command)
    env._last_force_budget = forces["force_budget"]
    x, env._y, z, vx, env._Vy, vz, env._bank, env._heading = state
    return x, z, vx, vz, eta, veta, forces['N_water'], forces['T'], forces['L'], forces['D']
