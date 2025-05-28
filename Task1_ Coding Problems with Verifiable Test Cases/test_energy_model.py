import numpy as np
import pytest
from urban_drive_simulation import run_urban_drive_simulation


def test_simulation_outputs_structure():
    """
    Ensure the simulation returns expected metrics in correct format.
    """
    result = run_urban_drive_simulation(seed=123)

    assert isinstance(result, dict)
    assert "total_distance_km" in result
    assert "energy_consumed_kwh" in result
    assert "energy_regen_kwh" in result
    assert "final_soc_kwh" in result


def test_soc_bounds():
    """
    Check that final SOC stays within battery capacity limits.
    """
    result = run_urban_drive_simulation(seed=456)

    final_soc = result["final_soc_kwh"]
    assert 0 <= final_soc <= 40  # kWh capacity


def test_energy_conservation():
    """
    Validate that regenerated energy does not exceed a physical threshold.
    """
    result = run_urban_drive_simulation(seed=789)

    regen = result["energy_regen_kwh"]
    consumed = result["energy_consumed_kwh"]
    assert regen < consumed  # Cannot regenerate more than consumed


def test_total_distance_static():
    """
    Distance should always match 25 km (as specified).
    """
    result = run_urban_drive_simulation(seed=999)
    assert result["total_distance_km"] == 25
