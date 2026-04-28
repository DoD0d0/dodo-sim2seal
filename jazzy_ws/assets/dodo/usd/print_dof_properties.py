from isaacsim.core.prims import SingleArticulation
prim_path = "/dodo_daimao"
prim = SingleArticulation(prim_path=prim_path, name= "dodo")
print(str(prim.dof_names))
print(prim.dof_properties)
