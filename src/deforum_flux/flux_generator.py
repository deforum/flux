# flux_generator.py
import time
import torch
from typing import Optional
from pydantic import BaseModel

from flux.modules.image_embedders import ReduxImageEncoder
from flux.sampling import (
    denoise,
    get_noise,
    get_schedule,
    prepare,
    prepare_kontext,
    prepare_redux,
    unpack
)
from flux.util import (
    check_onnx_access_for_trt,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

class FluxConfig(BaseModel):
    name: str = "flux-schnell"
    num_steps: Optional[int] = None
    offload: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trt: bool = False
    trt_transformer_precision: str = "bf16"
    trt_t5_precision: str = "bf16"
    trt_engines_dir: str = "checkpoints/trt_engines"
    trt_custom_onnx_paths: Optional[str] = None
    trt_image_height: Optional[int] = 1024
    trt_image_width: Optional[int] = 1024
    trt_batch_size: int = 1
    trt_timing_cache: Optional[str] = None
    trt_static_batch: bool = False
    trt_static_shape: bool = False

class FluxArgs(BaseModel):
    prompt: str
    img_cond_path: Optional[str] = None
    width: int = 1024
    height: int = 1024
    aspect_ratio: Optional[float] = None
    num_steps: Optional[int] = None
    guidance: Optional[float] = 3.5
    seed: Optional[int] = None

class FluxGenerator:
    def __init__(self, config: FluxConfig):
        self.config = config
        self.name = config.name
        self.offload = config.offload
        self.device = torch.device(config.device)

        if not config.trt:
            self.t5 = load_t5(device="cpu" if self.offload else self.device, max_length=256 if config.name == "flux-schnell" else 512)
            self.clip = load_clip(device="cpu" if self.offload else self.device)
            self.model = load_flow_model(config.name, device="cpu" if self.offload else self.device)
            self.ae = load_ae(config.name, device="cpu" if self.offload else self.device)
        else:
            # if TRT is enabled we must disable offloading moving TRT models to CPU
            self.offload = False

            # kontext repo does not use fp4, set it to fp4_svd32
            if config.name == "flux-dev-kontext" and config.trt_transformer_precision == "fp4":
                config.trt_transformer_precision = "fp4_svd32"

            # lazy import to make install optional
            from flux.trt.trt_manager import ModuleName, TRTManager

            # Check if we need ONNX model access (which requires authentication for FLUX models)
            onnx_dir = check_onnx_access_for_trt(config.name, config.trt_transformer_precision)
            trt_ctx_manager = TRTManager(
                trt_transformer_precision=config.trt_transformer_precision,
                trt_t5_precision=config.trt_t5_precision,
            )

            # kontext repo does have a VAE so we load dev version
            if config.name == "flux-dev-kontext":

                # Load TRT engines
                engines = trt_ctx_manager.load_engines(
                    model_name=config.name,
                    module_names={
                        ModuleName.CLIP,
                        ModuleName.TRANSFORMER,
                        ModuleName.T5,
                    },
                    engine_dir=config.trt_engines_dir,
                    custom_onnx_paths=onnx_dir or config.trt_custom_onnx_paths,
                    trt_image_height=config.trt_image_height,
                    trt_image_width=config.trt_image_width,
                    trt_batch_size=config.trt_batch_size,
                    trt_timing_cache=config.trt_timing_cache,
                    trt_static_batch=config.trt_static_batch,
                    trt_static_shape=config.trt_static_shape,
                )
                
                self.model = engines[ModuleName.TRANSFORMER].to(self.device)
                self.clip = engines[ModuleName.CLIP].to(self.device)
                self.t5 = engines[ModuleName.T5].to(self.device)
                self.ae = load_ae(config.name, self.device)
                
            else:

                # Load TRT engines
                engines = trt_ctx_manager.load_engines(
                    model_name=config.name,
                    module_names={
                        ModuleName.CLIP,
                        ModuleName.TRANSFORMER,
                        ModuleName.T5,
                        ModuleName.VAE,
                    },
                    engine_dir=config.trt_engines_dir,
                    custom_onnx_paths=onnx_dir or config.trt_custom_onnx_paths,
                    trt_image_height=config.trt_image_height,
                    trt_image_width=config.trt_image_width,
                    trt_batch_size=config.trt_batch_size,
                    trt_timing_cache=config.trt_timing_cache,
                    trt_static_batch=config.trt_static_batch,
                    trt_static_shape=config.trt_static_shape,
                )

                self.ae = engines[ModuleName.VAE].to(self.device)
                self.model = engines[ModuleName.TRANSFORMER].to(self.device)
                self.clip = engines[ModuleName.CLIP].to(self.device)
                self.t5 = engines[ModuleName.T5].to(self.device)

            # print all loaded engines
            print(f"self.ae: {self.ae}")
            print(f"self.model: {self.model}")
            print(f"self.clip: {self.clip}")
            print(f"self.t5: {self.t5}")

    @torch.inference_mode()
    def __call__(self, args: FluxArgs):
        # Use model-specific default if num_steps not provided
        if args.num_steps is None:
            args.num_steps = 4 if self.config.name == "flux-schnell" else 50

        # allow for packing and conversion to latent space
        height = 16 * (args.height // 16)
        width = 16 * (args.width // 16)

        # Set default seed if not provided
        if args.seed is None:
            rng = torch.Generator(device="cpu")
            args.seed = rng.seed()

        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=args.seed,
        )

        if self.offload:
            self.ae = self.ae.cpu()
            self.t5 = self.t5.cpu()
            self.clip = self.clip.cpu()
            self.model = self.model.cpu() 
            torch.cuda.empty_cache()

        if self.offload:
            self.t5 = self.t5.to(self.device)
            self.clip = self.clip.to(self.device)
            self.ae = self.ae.to(self.device)

        if self.name == "flux-dev-kontext":
            print(f"Image conditioning path: {args.img_cond_path}")
            inp, height, width = prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=args.prompt,
                ae=self.ae,
                img_cond_path=args.img_cond_path,
                target_width=args.width,
                target_height=args.height,
                bs=1,
                seed=args.seed,
                device=self.device,
            )
            # TODO check what this is
            inp.pop("img_cond_orig")
        else:
            inp = prepare(self.t5, self.clip, x, prompt=args.prompt)

        timesteps = get_schedule(args.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.ae = self.ae.cpu()
            self.t5 = self.t5.cpu()
            self.clip = self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(self.model, **inp, timesteps=timesteps, guidance=args.guidance)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model = self.model.cpu() 
            torch.cuda.empty_cache()
            self.ae.decoder = self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s")

        return x