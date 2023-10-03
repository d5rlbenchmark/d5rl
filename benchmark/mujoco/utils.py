from dm_control import composer, mjcf


def get_object_com(physics, body):
    return physics.bind(body.mjcf_model.find("body", "model")).subtree_com


def rescale_subtree(body, factor):
    for child in body.all_children():
        if getattr(child, "fromto", None) is not None:
            new_pos = factor * 0.5 * (child.fromto[3:] + child.fromto[:3])
            new_size = factor * 0.5 * (child.fromto[3:] - child.fromto[:3])
            child.fromto[:3] = new_pos - new_size
            child.fromto[3:] = new_pos + new_size
        if getattr(child, "pos", None) is not None:
            child.pos *= factor
        if getattr(child, "size", None) is not None:
            child.size *= factor
        if child.tag == "body" or child.tag == "worldbody":
            rescale_subtree(child, factor)


def rescale(root_body, factor):
    rescale_subtree(root_body, factor)

    mass_factor = factor**3
    for body in root_body.find_all("body"):
        inertial = getattr(body, "inertial", None)
        if inertial:
            inertial.mass *= mass_factor
    for geom in root_body.find_all("geom"):
        if geom.mass is not None:
            geom.mass *= mass_factor
        else:
            current_density = geom.density if geom.density is not None else 1000
            geom.density = current_density * mass_factor

    for mesh in root_body.asset.find_all("mesh"):
        mesh.scale = (factor, factor, factor)

    for joint in root_body.find_all("joint"):
        if joint.type == "slide":
            joint.range[:] *= factor


class XMLObject(composer.Entity):
    def _build(self, xml_path, scale: float = 1.0):
        self._mjcf_root = mjcf.from_path(xml_path)

        if scale != 1.0:
            rescale(self._mjcf_root, scale)

        for geom in self.mjcf_model.find_all("geom"):
            geom.solref = "0.002 1"
            geom.solimp = "0.95 0.99 0.001"
            geom.condim = "6"
            geom.friction = "1.0 0.1 0.001"

    @property
    def mjcf_model(self):
        return self._mjcf_root
