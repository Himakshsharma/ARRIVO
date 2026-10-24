# Data dictionary

## A. Raw columns (Kaggle "Delivery Truck Trips Data" layout — the synthetic twin uses the same names)

| Raw column | Meaning | Used? |
|---|---|---|
| `BookingID` | Unique booking identifier | ID only (deduplication) |
| `trip_start_date` | Dispatch timestamp | ✅ → `dispatch_time` |
| `Planned_ETA` | Planner's promised arrival time | ✅ → `planned_hours` (known at dispatch) |
| `actual_eta` / `trip_end_date` | Real arrival time | ⚠️ **target only** (never a feature) |
| `Origin_Location`, `Destination_Location` | City / place names | ✅ cleaned → `origin_city`, `destination_city` |
| `Org_lat_lon`, `Des_lat_lon` | "lat,lon" strings | ✅ → coordinates (weather, distance, region) |
| `TRANSPORTATION_DISTANCE_IN_KM` | Route distance | ✅ → `road_distance_km` |
| `vehicleType` | Truck type, e.g. "32 FT Multi-Axle" | ✅ |
| `Material Shipped` | Cargo category | ✅ → `material_type` |
| `supplierNameCode` / `supplierID` | Transport supplier | ✅ → `supplier_id` (via history feature) |
| `ontime`, `delay` | On-time / delay flags | ❌ leaky (post-delivery) |
| `Data_Ping_time`, `Current_Location`, `Curr_lat`, `Curr_lon` | Live GPS pings | ❌ leaky (only exist during/after the trip) |
| `Driver_Name`, `Driver_MobileNo` (Kaggle only) | Personal data | ❌ privacy — not loaded |

## B. Canonical schema (output of `src/data/load.py`)

| Column | Type | Description |
|---|---|---|
| `booking_id` | str | Unique booking ID |
| `dispatch_time` | datetime | When the truck left the origin |
| `trip_end_time` | datetime | When it arrived (used only to build targets) |
| `origin_city`, `destination_city` | str | Title-cased city names |
| `origin_lat`, `origin_lon`, `dest_lat`, `dest_lon` | float | Coordinates (degrees) |
| `origin_region`, `dest_region` | str | North / West / South / East / Central / Northeast (nearest hub) |
| `origin_is_hilly`, `dest_is_hilly` | 0/1 | Himalayan / north-east hill location |
| `road_distance_km` | float | Road distance (km) |
| `planned_hours` | float | Planned ETA − dispatch time (hours) |
| `vehicle_type` | str | Truck type (`Unknown` if missing) |
| `material_type` | str | Cargo category |
| `supplier_id` | str | Transport supplier code (`UNKNOWN` if missing) |

## C. Targets (created in `src/data/clean.py`)

| Column | Definition | Task |
|---|---|---|
| `actual_hours` | `trip_end_time − dispatch_time` in hours | **ETA regression target** |
| `delay_hours` | `actual_hours − planned_hours` | analysis only |
| `is_delayed` | `1` if `actual_hours > planned_hours + DELAY_GRACE_HOURS` (grace = 0 h, same as Kaggle's flag) | **Delay classification target** |

## D. Cleaning rules

| Rule | Reason |
|---|---|
| Drop duplicate `booking_id` (keep first) | GPS provider sometimes sends a booking twice |
| Drop rows with missing dispatch/arrival/planned time | no target can be computed |
| Drop trips with duration ≤ 0 or > 240 h, or average speed < 5 or > 90 km/h | timestamp errors, not genuine delays (Indian trucks are speed-governed at 80 km/h) |
| Missing distance → haversine × median detour factor | unbiased, cheap estimate |
| Missing vehicle / material / supplier → `Unknown` | keeps the row; missingness may carry signal |
| City names stripped, title-cased, state suffix removed | "DELHI ", "Delhi, Delhi" → "Delhi" |

## E. Synthetic twin – how it was generated (`generate_synthetic`)

* 27 Indian hubs with real coordinates; route choice ∝ hub size × e^(−distance/900 km).
* Vehicle speed 33–43 km/h by truck type (effective speed including stops); long hauls use bigger trucks.
* Transit time = distance ÷ speed × (1 + route congestion + supplier effect + cargo + hills + rain + festival + monsoon + rush hour − night) + supplier loading time + 1.5 h per region crossed, × log-normal noise, plus a 4% chance of a breakdown (exponential, mean 8 h). Ops improve slightly in 2025 (drift).
* Weather from regional monthly rainfall/temperature normals (IMD-like), deterministic per place and day.
* Planner ETA = distance ÷ (0.92 × nominal speed) + 3 h + 1 h per region crossed — ignores weather, festivals and route quirks (that is why trips run late).
* Injected problems: 3% missing vehicle type, 1% missing distance, 0.5% missing supplier, 5% messy city names, 0.4% negative durations, 0.3% 40-day trips, 1% duplicates.
