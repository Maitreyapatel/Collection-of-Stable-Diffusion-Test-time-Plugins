from dataclasses import dataclass, field
from utils.configs import AttendExciteConfig, LayoutGuidanceConfig, AttentionRefocusConfig
import sys
import pyrallis

import coloredlogs, logging

import torch
torch.autograd.set_detect_anomaly(True)

_EXPERIMENTS_ = {
    "aae": "Attend-and-Excite",
    "lg": "Layout-Guidance",
    "af": "Attention-Refocus",
}

def setup_logging():
    coloredlogs.install()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.NOTSET,
    )
    logging.root.setLevel(logging.NOTSET)

@dataclass
class TrainConfig:
    exp_name: str = None
    debugme: bool = False
    aae: AttendExciteConfig = field(default_factory=AttendExciteConfig)
    lg: LayoutGuidanceConfig = field(default_factory=LayoutGuidanceConfig)
    af: AttentionRefocusConfig = field(default_factory=AttentionRefocusConfig)

    def __post_init__(self):
        if self.exp_name not in list(_EXPERIMENTS_.keys()):
            raise NotImplementedError(f"{self.exp_name} is currencetly not supported.")

@pyrallis.wrap()
def main(cfg: TrainConfig):    
    if cfg.debugme:
        import debugpy
        strport = 4444
        debugpy.listen(strport)
        print(
            f"waiting for debugger on {strport}. Add the following to your launch.json and start the VSCode debugger with it:"
        )
        print(
            f'{{\n    "name": "Python: Attach",\n    "type": "python",\n    "request": "attach",\n    "connect": {{\n      "host": "localhost",\n      "port": {strport}\n    }}\n }}'
        )
        debugpy.wait_for_client()

    logging.info(f"We have initiated: {_EXPERIMENTS_[cfg.exp_name]}")
    if cfg.exp_name=="aae":
        from src.infer_attend_and_excite import RunAttendAndExcite
        RunAttendAndExcite(cfg.aae)
    elif cfg.exp_name=="lg":
        agg_methods = ["all_attention","aggregate_attention","aggregate_layer_attention"]
        for agg_method in agg_methods:
            cfg.lg.attention_aggregation_method = agg_method
            from src.infer_layout_guidance import RunLayoutGuidance
            RunLayoutGuidance(cfg.lg)
    elif cfg.exp_name=="af":
        from src.infer_attention_refocus import RunAttentionRefocus
        RunAttentionRefocus(cfg.af)
    else:
        raise NotImplementedError(f"{cfg.exp_name} is currencetly not supported.")

        # cfg.lg.attention_aggregation_method
        # from src.infer_layout_guidance import RunLayoutGuidance
        # RunLayoutGuidance(cfg.lg)
        
if __name__=="__main__":
    setup_logging()
    main()
