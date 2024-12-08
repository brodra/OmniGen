import os
import sys
import yaml
import numpy as np
from PIL import Image
from OmniGen import OmniGenPipeline

""" yaml
default:
  prompt: ""
  input_images: []
  height: 1024
  width: 1024
  max_input_image_size: 1024
  num_inference_steps: 50
  seed: 0
overrides:
  - seed: 0
    guidance_scale: 2.5
    img_guidance_scale: 2.5
  - seed: 1
    guidance_scale: 2.5
    img_guidance_scale: 2.5
"""


if __name__ == "__main__":
    pipe = OmniGenPipeline.from_pretrained(
        "Shitao/OmniGen-v1",
        quantization_config="bnb_4bit",
        low_cpu_mem_usage=True,
    )
    for config_file in sys.argv[1:]:
        config = yaml.safe_load(open(config_file))
        outdir = config.get("outdir", os.path.dirname(config_file))
        os.makedirs(outdir, exist_ok=True)
        default_config = config.get("default", {})
        for override in config.get("overrides", []):
            new_config = {**default_config, **override}
            outfile = "-".join([f"{k}={v:.1f}" for k, v in override.items()])
            outfile = os.path.join(outdir, outfile + ".png")
            if os.path.exists(outfile):
                continue
            print(outfile)
            result = pipe(**new_config)[0]
            result.save(outfile)
        