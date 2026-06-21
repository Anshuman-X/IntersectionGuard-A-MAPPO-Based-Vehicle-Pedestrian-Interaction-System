import os
import pandas as pd

def generate_vehicle_demand_and_types():
    # 1. Read calibration data from workspace data folder or D drive fallback
    calib_paths = [
        "data/sumo_vehicle_calibration.csv",
        r"D:\RESEARCH INTERNSHIP NITT\DATA SETS\cleaned_data\TRAINING_DATA\sumo_vehicle_calibration.csv"
    ]
    
    veh_df = None
    for path in calib_paths:
        if os.path.exists(path):
            try:
                veh_df = pd.read_csv(path)
                print(f"Loaded vehicle calibration from {path}")
                break
            except Exception as e:
                print(f"Error reading {path}: {e}")
                
    if veh_df is None:
        print("[Warning] Calibration data not found. Using default dataset calibration values.")
        veh_df = pd.DataFrame([
            {"vehicle_type": "car", "max_speed": 79.1981, "avg_speed": 2.0169, "flow_rate": 0.9987},
            {"vehicle_type": "motorcycle", "max_speed": 64.3907, "avg_speed": 2.5638, "flow_rate": 3.9100},
            {"vehicle_type": "auto", "max_speed": 42.9740, "avg_speed": 1.9692, "flow_rate": 1.0191},
            {"vehicle_type": "bus", "max_speed": 43.9376, "avg_speed": 1.5353, "flow_rate": 0.1343},
            {"vehicle_type": "truck", "max_speed": 9.1756, "avg_speed": 1.3742, "flow_rate": 0.0467},
            {"vehicle_type": "bicycle", "max_speed": 24.1780, "avg_speed": 4.6554, "flow_rate": 0.0380},
            {"vehicle_type": "cart", "max_speed": 15.0180, "avg_speed": 5.3082, "flow_rate": 0.0088},
            {"vehicle_type": "other", "max_speed": 17.9701, "avg_speed": 2.1546, "flow_rate": 0.0088}
        ])

    # 2. Write vehicle_types.add.xml
    types_path = "demand/vehicle_types.add.xml"
    
    # Map dataset type names to SUMO guiShapes, dimensions, and professional colors
    type_mappings = {
        "car": {"guiShape": "passenger", "color": "0.8,0.2,0.2", "accel": 2.0, "decel": 4.5, "sigma": 0.5, "length": 5.0},
        "motorcycle": {"guiShape": "motorcycle", "color": "0.2,0.8,0.2", "accel": 3.0, "decel": 5.0, "sigma": 0.9, "length": 2.0},
        "auto": {"guiShape": "delivery", "color": "0.8,0.8,0.2", "accel": 1.8, "decel": 4.0, "sigma": 0.8, "length": 3.0},
        "bus": {"guiShape": "bus", "color": "0.8,0.5,0.2", "accel": 1.2, "decel": 3.5, "sigma": 0.6, "length": 12.0},
        "truck": {"guiShape": "truck", "color": "0.5,0.5,0.5", "accel": 1.0, "decel": 3.0, "sigma": 0.6, "length": 10.0},
        "bicycle": {"guiShape": "bicycle", "color": "0.3,0.7,0.9", "accel": 0.8, "decel": 1.5, "sigma": 0.9, "length": 1.8},
        "cart": {"guiShape": "delivery", "color": "0.6,0.4,0.2", "accel": 0.5, "decel": 1.0, "sigma": 0.9, "length": 2.5},
        "other": {"guiShape": "passenger", "color": "0.7,0.3,0.7", "accel": 1.5, "decel": 3.5, "sigma": 0.7, "length": 4.5}
    }
    
    # Write calibrated types to additional file (NOTE: NO "av" type is added here)
    with open(types_path, "w") as f:
        f.write("<additional>\n\n")
        for _, row in veh_df.iterrows():
            vtype = row["vehicle_type"].strip()
            if vtype not in type_mappings:
                continue
            cfg = type_mappings[vtype]
            max_speed_ms = float(row["max_speed"]) / 3.6
            avg_speed_ms = float(row["avg_speed"]) / 3.6
            
            # Calibrate speedFactor using avg vs max ratio to represent speed distribution realistically
            speed_factor = min(1.0, max(0.2, avg_speed_ms / (max_speed_ms + 1e-6))) if max_speed_ms > 0 else 0.9
            
            f.write(f'    <!-- Calibrated {vtype} -->\n')
            f.write(f'    <vType id="{vtype}"\n')
            f.write(f'           accel="{cfg["accel"]}"\n')
            f.write(f'           decel="{cfg["decel"]}"\n')
            f.write(f'           sigma="{cfg["sigma"]}"\n')
            f.write(f'           length="{cfg["length"]}"\n')
            f.write(f'           maxSpeed="{max_speed_ms:.2f}"\n')
            f.write(f'           speedFactor="{speed_factor:.2f}"\n')
            f.write(f'           speedDev="0.1"\n')
            f.write(f'           color="{cfg["color"]}"\n')
            f.write(f'           guiShape="{cfg["guiShape"]}"/>\n\n')
        f.write("</additional>\n")
    print(f"Generated calibrated vehicle types (av type removed) in {types_path}")

    # 3. Write t_routes.rou.xml with calibrated flows (NOTE: NO "av" flows are written here)
    routes_path = "demand/t_routes.rou.xml"
    with open(routes_path, "w") as f:
        f.write("<routes>\n\n")
        f.write('    <!-- Calibrated Routes -->\n')
        f.write('    <route id="west_to_east" edges="west_center center_east"/>\n')
        f.write('    <route id="east_to_west" edges="east_center center_west"/>\n')
        f.write('    <route id="north_to_west" edges="north_center center_west"/>\n\n')
        
        # Write flows for each calibrated type
        for _, row in veh_df.iterrows():
            vtype = row["vehicle_type"].strip()
            if vtype not in type_mappings:
                continue
            
            # Calibrate flow rate (vehsPerHour = flow_rate * 60)
            total_flow = max(1.0, float(row["flow_rate"]) * 60.0)
            
            # Distribute flows across the routes:
            # West-to-East: 40%
            # East-to-West: 40%
            # North-to-West: 20%
            flow_we = max(1, int(total_flow * 0.4))
            flow_ew = max(1, int(total_flow * 0.4))
            flow_nw = max(1, int(total_flow * 0.2))
            
            f.write(f'    <!-- {vtype.capitalize()} flows -->\n')
            f.write(f'    <flow id="{vtype}_flow_we"\n')
            f.write(f'          route="west_to_east"\n')
            f.write(f'          begin="0"\n')
            f.write(f'          end="1000"\n')
            f.write(f'          vehsPerHour="{flow_we}"\n')
            f.write(f'          type="{vtype}"/>\n\n')
            
            f.write(f'    <flow id="{vtype}_flow_ew"\n')
            f.write(f'          route="east_to_west"\n')
            f.write(f'          begin="0"\n')
            f.write(f'          end="1000"\n')
            f.write(f'          vehsPerHour="{flow_ew}"\n')
            f.write(f'          type="{vtype}"/>\n\n')
            
            if flow_nw > 0:
                f.write(f'    <flow id="{vtype}_flow_nw"\n')
                f.write(f'          route="north_to_west"\n')
                f.write(f'          begin="0"\n')
                f.write(f'          end="1000"\n')
                f.write(f'          vehsPerHour="{flow_nw}"\n')
                f.write(f'          type="{vtype}"/>\n\n')
                
        f.write("</routes>\n")
    print(f"Generated calibrated routes (av flows removed) in {routes_path}")

if __name__ == "__main__":
    generate_vehicle_demand_and_types()