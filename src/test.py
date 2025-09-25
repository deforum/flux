from deforum_flux import FluxGenerator, FluxConfig, FluxArgs, save_image

config = FluxConfig(
    name="flux-dev-kontext",
    offload=True
)

flux = FluxGenerator(config)

args = FluxArgs(
    prompt="add a unicorn",
    img_kontext="input.png",
    width=1392,
    height=752,
    num_steps=25
)

print(args)
image = flux(args)
save_image("output.jpg", image)