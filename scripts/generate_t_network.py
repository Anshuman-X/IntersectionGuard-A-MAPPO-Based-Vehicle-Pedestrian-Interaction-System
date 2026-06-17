# T-Intersection with sidewalks

nodes = """<nodes>
    <node id="west" x="-100" y="0" type="priority"/>
    <node id="center" x="0" y="0" type="priority"/>
    <node id="east" x="100" y="0" type="priority"/>
    <node id="north" x="0" y="100" type="priority"/>
</nodes>
"""

edges = """<edges>

    <!-- West -->
    <edge id="west_center"
          from="west"
          to="center"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

    <edge id="center_west"
          from="center"
          to="west"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

    <!-- East -->
    <edge id="east_center"
          from="east"
          to="center"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

    <edge id="center_east"
          from="center"
          to="east"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

    <!-- North -->
    <edge id="north_center"
          from="north"
          to="center"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

    <edge id="center_north"
          from="center"
          to="north"
          numLanes="1"
          speed="13.9"
          sidewalkWidth="2"/>

</edges>
"""

with open("network/t_nodes.nod.xml", "w") as f:
    f.write(nodes)

with open("network/t_edges.edg.xml", "w") as f:
    f.write(edges)

print("Pedestrian-ready network created.")