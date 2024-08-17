import torch
from torch.nn import Parameter
from transformers import AutoModel, AutoTokenizer

from .config import GlobalConfig
from .data import get_dataloader
from .track_prof import MlTrackContext
from .trainer import Trainer


def run():
    config = GlobalConfig()
    print(config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    model = AutoModel.from_pretrained(
        config.model_path, trust_remote_code=True
    ).to("cuda")
    model.scale_fac_parameter = Parameter(torch.zeros(1).to("cuda"))

    train_loader, valid_loader = get_dataloader(
        **config.data_config.export("kwargs"),
        tokenizer=tokenizer
    )

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        config.epochs,
        config.optimizer_config,
        scale_fac_type=config.scale_fac_type
    )

    with MlTrackContext(config, track=config.track):
        model = trainer.train(
            dev_test=config.dev_test,
            track=config.track,
            profiler_config=config.profiler_config
        )

    if config.save:
        config.save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.save_path)
        tokenizer.save_pretrained(config.save_path)
        print(f"Model saved at {config.save_path}")


if __name__ == "__main__":
    run()
