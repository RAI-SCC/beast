# Dataset variables
## Order
In general the order in the data arrays will be:
- Surface variable 1
- Surface variable 2
- ...
- Surface variable N
- Pressure variable 1 at pressure level 1
- Pressure variable 1 at pressure level 2
- ...
- Pressure variable 1 at pressure level M
- ...
- Pressure variable K at pressure level M

## Surface variables (in order)
1. Mean sea level pressure in Pa (mean_sea_level_pressure)
2. U-component of wind velocity at 10m in m/s (10m_u_component_of_wind)
3. V-component of wind velocity at 10m in m/s (10m_v_component_of_wind)
4. Temperature at 2m in K (2m_temperature)
5. Sea surface temperature in K (sea_surface_temperature)
6. Total precipitation in mm (?) (total_precipitation)

## Pressure variables (in order)
1. Geopotential in m^2 / s^2 (geopotential)
2. Specific humidity, unitless (or kg / kg according to ERA5) (specific_humidity)
3. Temperature in K (temperature)
4. U-component of wind velocity in m/s (u_component_of_wind)
5. V-component of wind velocity in m/s (v_component_of_wind)
6. W-component of wind velocity (vertical winds) in Pa/s (due to pressure based coordinates, < 0 means upward motion) (vertical_velocity)

## Pressure levels in hPa (in order)
1. 1000
2. 925
3. 850
4. 700
5. 600
6. 500
7. 400
8. 300
9. 250
10. 200
11. 150
12. 100
13. 50
