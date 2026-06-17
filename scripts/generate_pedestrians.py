import random

def generate_pedestrian_demand(file_path, total_time=1000, num_pedestrians=150, seed=42):
    random.seed(seed)
    
    # Define pedestrian OD paths that require crossing the junction
    # Path name -> sequence of edges
    paths = [
        # Crosses West leg
        ("west_to_west", ["west_center", "center_west"]),
        # Crosses East leg
        ("east_to_east", ["east_center", "center_east"]),
        # Crosses North leg
        ("north_to_north", ["north_center", "center_north"]),
        # Crosses West leg and North leg
        ("west_to_north", ["west_center", "center_north"]),
        # Crosses East leg and North leg
        ("east_to_north", ["east_center", "center_north"]),
        # Crosses North leg and West leg
        ("north_to_west", ["north_center", "center_west"]),
        # Crosses North leg and East leg
        ("north_to_east", ["north_center", "center_east"]),
    ]
    
    # Generate departure times (evenly distributed with some randomness)
    depart_times = sorted([random.uniform(5, total_time - 50) for _ in range(num_pedestrians)])
    
    with open(file_path, "w") as f:
        f.write("<routes>\n\n")
        
        # Define a pedestrian type with realistic speed and dimensions
        # Average walking speed is 1.34 m/s (standard deviation ~0.26 m/s)
        f.write("    <!-- Pedestrian Type Definition -->\n")
        f.write('    <vType id="pedestrian_type" vClass="pedestrian" speedDev="0.2" length="0.5" width="0.5" minGap="0.2"/>\n\n')
        
        for i, depart in enumerate(depart_times):
            path_name, edges = random.choice(paths)
            edges_str = " ".join(edges)
            
            f.write(f'    <person id="ped_{i}" type="pedestrian_type" depart="{depart:.2f}">\n')
            f.write(f'        <walk edges="{edges_str}"/>\n')
            f.write('    </person>\n\n')
            
        f.write("</routes>\n")
        
    print(f"Successfully generated {num_pedestrians} pedestrians in {file_path}")

if __name__ == "__main__":
    generate_pedestrian_demand("demand/pedestrians.rou.xml")
