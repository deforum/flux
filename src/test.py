from deforum_flux import FluxGenerator, FluxConfig, FluxArgs, save_image

config = FluxConfig(
    name="flux-dev",
    offload=True
)

flux = FluxGenerator(config)

args = FluxArgs(
    prompt="a detailed portrait of a person", 
    # img_cond="input.jpg",
    init_image="input.jpg",
    strength=0.025,
    width=1024,
    height=1024,
    num_steps=25,
    guidance=3.5,
)
print(args)
image = flux(args)
save_image("output.jpg", image)