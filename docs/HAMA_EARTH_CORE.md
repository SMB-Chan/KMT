# HAMA Earth Core v0.1

HAMA Earth Core is the first product-independent layer inside KMT. It turns the existing coastal flight simulation into the beginning of a reusable Earth-environment sandbox.

## Contract

- `EarthPatchConfig`: geographic origin, local simulation extent, absolute start time, deterministic seed.
- `EnvironmentProvider`: answers environmental conditions at a local `(x, y, z)` and simulation time.
- `EnvironmentSnapshot`: air temperature, pressure, density, 3-D wind, surface elevation and water depth.
- `SimulationObject`: any product or concept that can be placed in the world, advanced by a fixed timestep, and serialized.
- `EarthWorld`: owns simulation time, environment sampling and object ordering. Product-specific physics do not belong here.
- `ScenarioSpec`: JSON contract that describes a world, environment provider, run conditions and objects.

Coordinates are local tangent-plane metres: **x north, y east, z up**. The geographic origin is metadata in v0.1; geodetic conversion is intentionally deferred until terrain/bathymetry providers are introduced.

## Legacy KMT bridge

`KmtCoastalEnvironment` adapts the existing `Atmosphere` and `Ocean` models without modifying them. The existing PM ocean is currently 1-D, so `x` drives wave elevation and `y` is ignored. This is declared in snapshot metadata rather than hidden.

Run the reference scenario:

```bash
python3 -m hama.run_scenario examples/scenarios/coastal_reference.json
```

The first reference object is `EnvironmentalProbe`. It has no aircraft assumptions and demonstrates that a generic object can consume the Earth contract.

## v0.1 boundary

This commit establishes interfaces, not fidelity claims. It does **not** yet provide terrain, bathymetry grids, 2-D waves, weather reanalysis, precipitation, solar radiation, soil, hydrology, structural FEA or CFD. Those should be added as replaceable providers so experiments retain the same scenario and object contracts.

## Next milestones

1. Add geodetic/local coordinate conversion and terrain/bathymetry providers.
2. Add a 2-D coastal ocean provider and real-data provenance fields.
3. Wrap the current KMT flying boat as the first non-probe `SimulationObject`.
4. Add uncertainty/provenance to every environmental field.
5. Expose the same scenario/world operations as structured tools for LLM agents and the human UI.
