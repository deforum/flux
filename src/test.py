from deforum_flux import FluxGenerator, FluxConfig, FluxArgs, save_image

config = FluxConfig()
flux = FluxGenerator(config)
args = FluxArgs(prompt="A beautiful landscape with mountains and a river")
image = flux(args)
save_image("output.jpg", image)