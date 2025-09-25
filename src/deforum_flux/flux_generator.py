# flux_generator.py
import time
import torch
from einops import rearrange, repeat

from flux.sampling import (
    denoise,
    denoise_rf,
    get_noise,
    get_schedule,
    prepare,
    prepare_kontext,
    prepare_redux,
    unpack,
    encode_image
)
from flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_redux
)

from .flux_config import FluxConfig
from .flux_args import FluxArgs

class FluxGenerator:
    def __init__(self, config: FluxConfig):
        self.config = config
        self.name = config.name
        self.offload = config.offload
        self.device = torch.device(config.device)
        self.t5 = load_t5(device="cpu" if self.offload else self.device, max_length=256 if config.name == "flux-schnell" else 512)
        self.clip = load_clip(device="cpu" if self.offload else self.device)
        self.model = load_flow_model(config.name, device="cpu" if self.offload else self.device)
        self.ae = load_ae(config.name, device="cpu" if self.offload else self.device)
        self.redux = load_redux(device="cpu" if self.offload else self.device)

    @torch.inference_mode()
    def __call__(self, args: FluxArgs):

        t0 = time.perf_counter()
        x = get_noise(1, args.height, args.width, device=self.device, dtype=torch.bfloat16, seed=args.seed)

        if self.offload:
            self.ae = self.ae.cpu()
            self.t5 = self.t5.cpu()
            self.clip = self.clip.cpu()
            self.model = self.model.cpu()
            self.redux = self.redux.cpu()
            torch.cuda.empty_cache()

        if self.offload:
            self.t5 = self.t5.to(self.device)
            self.clip = self.clip.to(self.device)
            self.ae = self.ae.to(self.device)

        # prepare inputs
        if self.name == "flux-dev-kontext" and args.img_kontext is not None:
            inp = prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=args.prompt,
                ae=self.ae,
                img_cond=args.img_kontext,
                target_width=args.width,
                target_height=args.height,
                bs=1,
                seed=args.seed,
                device=self.device,
            )
        else:
            # handle redux image embedding
            if args.img_redux is not None:
                if self.offload:
                    self.redux = self.redux.to(self.device)
                    torch.cuda.empty_cache()
                inp = prepare_redux(
                    self.t5,
                    self.clip,
                    x,
                    prompt=args.prompt,
                    encoder=self.redux,
                    img_cond=args.img_redux,
                    device=self.device,
                    redux_strength=args.redux_strength,
                    redux_mask=args.redux_mask,
                    seed=args.seed,
                )
            else:
                inp = prepare(self.t5, self.clip, x, prompt=args.prompt)

        timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        if self.offload:
            self.ae = self.ae.cpu()
            self.t5 = self.t5.cpu()
            self.clip = self.clip.cpu()
            self.redux = self.redux.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        if args.init_image is not None:
            encoded_img = encode_image(self.ae, args.init_image, args.height, args.width, self.device)
            encoded_img = rearrange(encoded_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            encoded_img = encoded_img.to(self.device)
            x = denoise_rf(
                self.model,
                **inp,
                encoded_img=encoded_img,
                timesteps=timesteps,
                guidance=args.guidance,
                gamma=args.gamma,
                eta=args.eta,
                start_timestep=args.start_timestep,
                stop_timestep=args.stop_timestep
            )
        else:
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=args.guidance)

        if self.offload:
            self.model = self.model.cpu() 
            torch.cuda.empty_cache()
            self.ae.decoder = self.ae.decoder.to(x.device)

        x = unpack(x.float(), args.height, args.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s")

        return x