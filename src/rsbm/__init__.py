from . import bridges, normalizations, reflectors, sdes


# putting this here for now
def create_sde_and_bridge(
    sde_conf, normalization: normalizations.Normalization
) -> tuple[sdes.SDE, bridges.BrownianBridge]:
    if sde_conf.type == "brownian":
        sde = sdes.SDE(sigma=sde_conf.sigma, euler_steps=sde_conf.euler_steps)
        bridge = bridges.BrownianBridge(sigma=sde_conf.sigma)
    elif sde_conf.type == "reflected":
        reflector = reflectors.Reflector(
            n_reflections=sde_conf.n_reflections, normalization=normalization
        )
        sde = sdes.ReflectedSDE(
            sigma=sde_conf.sigma, euler_steps=sde_conf.euler_steps, reflector=reflector
        )
        bridge = bridges.ReflectedBrownianBridge(
            sigma=sde_conf.sigma, reflector=reflector
        )
    else:
        raise ValueError(f"Unknown SDE type: {sde_conf.type}")
    return sde, bridge
