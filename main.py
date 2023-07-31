from dataclasses import dataclass, field
import sys
import pyrallis
import coloredlogs, logging

from utils.configs import DivideAndConquerConfig, AttendExciteConfig, LayoutGuidanceConfig, AttentionRefocusConfig, TrainerConfig, TestConfig

import torch
torch.autograd.set_detect_anomaly(True)

_EXPERIMENTS_ = {
    "aae": "Attend-and-Excite",
    "lg": "Layout-Guidance",
    "af": "Attention-Refocus",
    "dac": "Divide-And-Conquer",
    "train": "Training Model",
    "test": "Testing the provided model"
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
    dac: DivideAndConquerConfig = field(default_factory=DivideAndConquerConfig)
    aae: AttendExciteConfig = field(default_factory=AttendExciteConfig)
    lg: LayoutGuidanceConfig = field(default_factory=LayoutGuidanceConfig)
    af: AttentionRefocusConfig = field(default_factory=AttentionRefocusConfig)
    train: TrainerConfig = field(default_factory=TrainerConfig)
    test: TestConfig = field(default_factory=TestConfig)

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
        from src.infer_layout_guidance import RunLayoutGuidance
        RunLayoutGuidance(cfg.lg)
    elif cfg.exp_name=="af":
        from src.infer_attention_refocus import RunAttentionRefocus
        RunAttentionRefocus(cfg.af)
    elif cfg.exp_name=="dac":
        from src.infer_divide_and_conquer import RunDivideAndConquer
        RunDivideAndConquer(cfg.dac)

    elif cfg.exp_name=="train":
        from src.trainer import run_experiment
        run_experiment(cfg.train)
    elif cfg.exp_name=="test":
        from src.test import run_inference
        run_inference(cfg.test)
    else:
        raise NotImplementedError(f"{cfg.exp_name} is currencetly not supported.")
        
if __name__=="__main__":
    setup_logging()
    main()
