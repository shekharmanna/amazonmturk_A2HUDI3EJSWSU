import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt


def run_urban_drive_simulation(seed: int = 42) -> dict:
    """
    Simulates an urban driving cycle for a 1200 kg hybrid vehicle and estimates
    energy consumption, regenerative braking energy recovery, and final SOC.
    
    Parameters:
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Summary containing total distance, energy consumed, regen energy, and final SOC.
    """

    # Vehicle and environment constants
    vehicle_mass = 1200  # in kg
    gravity = 9.81  # m/s^2
    rolling_resistance_coeff = 0.015
    air_density = 1.225  # kg/m^3
    drag_coefficient = 0.29
    frontal_area = 2.2  # m^2

    # Powertrain and energy system
    regen_efficiency = 0.6  # 60% regen braking efficiency
    motor_efficiency = 0.9  # 90% drive efficiency
    battery_capacity_kwh = 40  # Battery capacity in kWh
    soc_start = 0.9  # Start with 90% SOC
    hvac_load_kw = 1.5  # HVAC load in kW

    # Drive cycle and simulation configuration
    distance_total_km = 25
    drive_cycle_duration = 3600  # 1 hour = 3600 seconds
    time_step = 1  # 1 second resolution
    np.random.seed(seed)

    # Time series and synthetic speed profile (stop-go urban pattern)
    time_series = np.arange(0, drive_cycle_duration, time_step)
    speed_profile = np.clip(
        np.cumsum(np.random.normal(0, 0.3, len(time_series))), 0, 15
    )  # Speed in m/s

    acceleration = np.gradient(speed_profile, time_step)

    # Initialize metrics
    total_energy_consumed_kwh = 0.0
    total_energy_regen_kwh = 0.0
    soc = soc_start * battery_capacity_kwh

    # Energy computation over the time series
    for v, a in zip(speed_profile, acceleration):
        # Forces
        f_rolling = vehicle_mass * gravity * rolling_resistance_coeff
        f_aero = 0.5 * air_density * drag_coefficient * frontal_area * v**2
        f_inertia = vehicle_mass * a

        total_force = f_rolling + f_aero + f_inertia
        power_watt = total_force * v

        # Add HVAC load
        power_watt += hvac_load_kw * 1000  # Convert kW to W

        energy_joules = power_watt * time_step

        if energy_joules > 0:
            # Discharging energy
            energy_kwh = energy_joules / 3.6e6 / motor_efficiency
            total_energy_consumed_kwh += energy_kwh
            soc -= energy_kwh
        else:
            # Regenerative braking
            energy_kwh = -energy_joules / 3.6e6 * regen_efficiency
            total_energy_regen_kwh += energy_kwh
            soc += energy_kwh

        # Clamp SOC to valid range
        soc = max(0, min(battery_capacity_kwh, soc))

    # Optional visualization (can be commented out for headless testing)
    plt.figure(figsize=(10, 3))
    plt.plot(time_series, speed_profile * 3.6)  # Convert m/s to km/h
    plt.title("Urban Drive Cycle - Speed Profile")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "total_distance_km": distance_total_km,
        "energy_consumed_kwh": round(total_energy_consumed_kwh, 2),
        "energy_regen_kwh": round(total_energy_regen_kwh, 2),
        "final_soc_kwh": round(soc, 2)
    }


# Run simulation if executed directly
if __name__ == "__main__":
    results = run_urban_drive_simulation()
    print("Simulation Results:")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
