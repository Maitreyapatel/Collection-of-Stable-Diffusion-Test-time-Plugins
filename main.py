from dataclasses import dataclass, field
from utils.configs import AttendExciteConfig

import pyrallis

import coloredlogs, logging

_EXPERIMENTS_ = {
    "aae": "Attend-and-Excite"
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
    aae: AttendExciteConfig = field(default_factory=AttendExciteConfig)

    def __post_init__(self):
        if self.exp_name not in list(_EXPERIMENTS_.keys()):
            raise NotImplementedError(f"{self.exp_name} is currencetly not supported.")

@pyrallis.wrap()
def main(cfg: TrainConfig):
    logging.info(f"We have initiated: {_EXPERIMENTS_[cfg.exp_name]}")
    if cfg.exp_name=="aae":
        from src.infer_attend_and_excite import RunAttendAndExcite
        RunAttendAndExcite(cfg.aae)
        
if __name__=="__main__":
    setup_logging()
    main()
