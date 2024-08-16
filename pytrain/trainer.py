import mlflow
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.profiler import profile
from torch.utils.data import DataLoader
from torchmetrics.regression import CosineSimilarity
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm

from .config import OptimizerConfig, ProfilerConfig
from .optimizer import get_optimizer_scheduler
from .track_prof import TorchProfilerContext


class Trainer:
    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int,
        optimizer_config: OptimizerConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = epochs
        self.optimizer, self.lr_scheduler = get_optimizer_scheduler(
            model,
            total_train_step=len(train_loader) * epochs,
            **optimizer_config.export()
        )
        self.step = 0

    def infer(self, model_inp: BatchEncoding) -> Tensor:
        model_inp = model_inp.to("cuda")
        embedded_sent = self.model(**model_inp)["pooler_output"]

        embedded_question = embedded_sent[::2]
        embedded_context = embedded_sent[1::2]
        similarities_matrix = embedded_question @ embedded_context.T
        return similarities_matrix * self.model.scale_fac_parameter

    def train_loop(
        self,
        dev_test: bool = False,
        track: bool = False,
        profiler: profile | None = None
    ) -> Tensor:
        self.model.train()
        list_loss = Tensor([]).to("cuda")
        loop = tqdm(self.train_loader, ascii=True)
        criterion = CrossEntropyLoss()
        labels = torch.arange(0, self.train_loader.batch_size, device="cuda")

        for i, model_inp in enumerate(loop):
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            similarities_matrix = self.infer(model_inp)

            loss_question = criterion(similarities_matrix, labels)
            loss_context = criterion(similarities_matrix.T, labels)
            loss = (loss_question + loss_context) / 2

            loss.backward()
            self.optimizer.step()

            list_loss = torch.cat((list_loss, loss.detach().data.view(1)))
            avg_loss = list_loss.mean().item()

            if track:
                mlflow.log_metrics(
                    {
                        "loss": loss.item(),
                        "avg_loss": avg_loss,
                        "scale_factor": self.model.scale_fac_parameter.item()
                    },
                    step=self.step,
                )
            loop.set_postfix(loss=loss.item(), avg_loss=avg_loss)
            self.step += 1

            if profiler is not None:
                profiler.step()

            if dev_test and i == 20:
                loop.close()
                print(
                    "Max memory allocated:",
                    torch.cuda.max_memory_allocated(device='cuda')/(1024**3)
                )
                break

        return list_loss

    def eval_infer(self, model_inp: BatchEncoding) -> tuple[Tensor, Tensor, Tensor]:
        model_inp = model_inp.to("cuda")
        embedded_sent = self.model(**model_inp)["pooler_output"]

        embedded_question = embedded_sent[::3]
        embedded_ans_text = embedded_sent[1::3]
        embedded_ans_md = embedded_sent[2::3]
        return embedded_question, embedded_ans_text, embedded_ans_md

    @torch.no_grad()
    def valid_loop(
        self, dev_test: bool = False, track: bool = False
    ) -> tuple[CosineSimilarity, CosineSimilarity]:
        self.model.eval()
        loop = tqdm(self.valid_loader, ascii=True)
        metric_text = CosineSimilarity(reduction='mean').to("cuda")
        metric_md = CosineSimilarity(reduction='mean').to("cuda")

        for i, model_inp in enumerate(loop):
            embedded_question, embedded_ans_text, embedded_ans_md = self.eval_infer(
                model_inp
            )
            score_text = metric_text(embedded_question, embedded_ans_text)
            score_md = metric_md(embedded_question, embedded_ans_md)

            loop.set_postfix(
                batch_score_text=score_text.cpu().item(),
                batch_score_md=score_md.cpu().item(),
                avg_score_text=metric_text.compute().cpu().item(),
                avg_score_md=metric_md.compute().cpu().item(),
            )

            if dev_test and i == 20:
                loop.close()
                break

        if track:
            mlflow.log_metrics(
                {
                    "avg_cossim_text": metric_text.compute().cpu().item(),
                    "avg_cossim_md": metric_md.compute().cpu().item(),
                },
                step=self.step
            )

        return metric_text, metric_md

    def train(
        self,
        epochs: int | None = None,
        dev_test: bool = False,
        track: bool = False,
        profiler_config: ProfilerConfig | None = None
    ) -> Module:
        if epochs is None:
            epochs = self.epochs

        metric_text, metric_md = self.valid_loop(dev_test=dev_test, track=track)
        print(
            f"Initial cos similarity for text: {metric_text.compute().cpu().item()}",
            f"Initial cos similarity for markdown: {metric_md.compute().cpu().item()}",
            sep="\n"
        )

        for epoch in range(epochs):
            print(
                "*"*40, f"Epoch {epoch+1}/{epochs}", "*"*40
            )

            with TorchProfilerContext(
                **({} if profiler_config is None else profiler_config.export()),
            ) as profiler:
                list_loss = self.train_loop(
                    dev_test=dev_test, track=track, profiler=profiler
                )

            metric_text, metric_md = self.valid_loop(dev_test=dev_test, track=track)

            print(
                f"average loss: {list_loss.mean().item()} |",
                f"cossim text score: {metric_text.compute().cpu().item()} |",
                f"cossim md score: {metric_md.compute().cpu().item()}",
            )

        return self.model
