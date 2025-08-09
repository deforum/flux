from deforum_flux import FluxGenerator, FluxConfig, FluxArgs, save_image

config = FluxConfig(
    name="flux-dev",
    offload=True
)

flux = FluxGenerator(config)

args = FluxArgs(
    prompt="make it red", 
    img_cond="input.jpg",
    width=1024,
    height=1024,
    num_steps=50,
    guidance=3.5,
    seed=42
)
print(args)
image = flux(args)
save_image("output.jpg", image)