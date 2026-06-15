import xml.etree.ElementTree as ET

routes = """
<routes>

    <route id="west_to_east"
           edges="west_center center_east"/>

    <route id="east_to_west"
           edges="east_center center_west"/>

    <route id="north_to_west"
           edges="north_center center_west"/>

    <flow id="car_flow"
          route="west_to_east"
          begin="0"
          end="300"
          vehsPerHour="300"
          type="car"/>

    <flow id="bike_flow"
          route="east_to_west"
          begin="0"
          end="300"
          vehsPerHour="500"
          type="motorcycle"/>

    <flow id="auto_flow"
          route="north_to_west"
          begin="0"
          end="300"
          vehsPerHour="150"
          type="autorickshaw"/>

</routes>
"""

with open("demand/t_routes.rou.xml", "w") as f:
    f.write(routes)

print("Route file updated.")