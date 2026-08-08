from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Protocol, TypedDict


@dataclass
class SamplingParams:
    temperature: float = 0.6
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_sequence_ids = count()


@dataclass
class Sequence:
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    seq_id: int = field(
        default_factory=lambda: next(_sequence_ids),
        init=False,
    )
    completion_token_ids: list[int] = field(
        default_factory=list,
        init=False,
    )
    status: SequenceStatus = field(
        default=SequenceStatus.WAITING,
        init=False,
    )

    def __post_init__(self) -> None:
        self.prompt_token_ids = list(self.prompt_token_ids)

        if not self.prompt_token_ids:
            raise ValueError("prompt must contain at least one token")

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.completion_token_ids

    @property
    def is_finished(self) -> bool:
        return self.status is SequenceStatus.FINISHED

    def append_token(self, token_id: int) -> None:
        self.completion_token_ids.append(token_id)


class SchedulerProtocol(Protocol):
    def add(self, sequence: Sequence) -> None:
        ...

    def schedule(self) -> tuple[list[Sequence], bool]:
        ...

    def postprocess(
        self,
        sequences: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> None:
        ...

    def is_finished(self) -> bool:
        ...


class ModelRunnerProtocol(Protocol):
    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        ...

    def close(self) -> None:
        ...


class TokenizerProtocol(Protocol):
    eos_token_id: int

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...


class GenerationOutput(TypedDict):
    text: str
    token_ids: list[int]


class LLMEngine:
    """Small dependency-injected LLM control plane."""

    def __init__(
        self,
        tokenizer: TokenizerProtocol,
        scheduler: SchedulerProtocol,
        model_runner: ModelRunnerProtocol,
        *,
        max_model_len: int = 4096,
    ) -> None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")

        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.model_runner = model_runner
        self.max_model_len = max_model_len
        self._closed = False

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> int:
        if self._closed:
            raise RuntimeError("engine is closed")

        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = list(prompt)

        requested_length = (
            len(token_ids) + sampling_params.max_tokens
        )

        if requested_length > self.max_model_len:
            raise ValueError(
                f"request needs {requested_length} tokens, "
                f"but max_model_len={self.max_model_len}"
            )

        sequence = Sequence(token_ids, sampling_params)
        self.scheduler.add(sequence)

        return sequence.seq_id

    def step(self) -> list[Sequence]:
        if self.scheduler.is_finished():
            return []

        sequences, is_prefill = self.scheduler.schedule()

        if not sequences:
            raise RuntimeError(
                "scheduler returned an empty batch while work remains"
            )

        token_ids = self.model_runner.run(
            sequences,
            is_prefill,
        )

        if len(token_ids) != len(sequences):
            raise RuntimeError(
                "model runner must return exactly one token "
                "per scheduled sequence"
            )

        self.scheduler.postprocess(
            sequences,
            token_ids,
            is_prefill,
        )

        return [
            sequence
            for sequence in sequences
            if sequence.is_finished
        ]

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
    ) -> list[GenerationOutput]:
        if not self.scheduler.is_finished():
            raise RuntimeError(
                "this simplified engine does not allow "
                "overlapping generate calls"
            )

        if isinstance(sampling_params, SamplingParams):
            params = [sampling_params] * len(prompts)
        else:
            params = list(sampling_params)

            if len(params) != len(prompts):
                raise ValueError(
                    "prompts and sampling_params must have equal lengths"
                )

        request_ids = [
            self.add_request(prompt, params[index])
            for index, prompt in enumerate(prompts)
        ]

        completed: dict[int, list[int]] = {}

        while not self.scheduler.is_finished():
            for sequence in self.step():
                completed[sequence.seq_id] = list(
                    sequence.completion_token_ids
                )

        return [
            {
                "text": self.tokenizer.decode(completed[seq_id]),
                "token_ids": completed[seq_id],
            }
            for seq_id in request_ids
        ]

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self.model_runner.close()

    def __enter__(self) -> LLMEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SimpleScheduler:
    """Simple prefill/decode scheduler for tests."""

    def __init__(
        self,
        eos_token_id: int,
        max_num_sequences: int = 8,
    ) -> None:
        if max_num_sequences <= 0:
            raise ValueError(
                "max_num_sequences must be positive"
            )

        self.eos_token_id = eos_token_id
        self.max_num_sequences = max_num_sequences

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, sequence: Sequence) -> None:
        self.waiting.append(sequence)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        # Prefill new requests first.
        if self.waiting:
            batch: list[Sequence] = []

            while (
                self.waiting
                and len(batch) < self.max_num_sequences
            ):
                sequence = self.waiting.popleft()
                sequence.status = SequenceStatus.RUNNING

                self.running.append(sequence)
                batch.append(sequence)

            return batch, True

        # Decode running requests fairly.
        batch = []

        for _ in range(
            min(len(self.running), self.max_num_sequences)
        ):
            sequence = self.running.popleft()
            self.running.append(sequence)
            batch.append(sequence)

        return batch, False

    def postprocess(
        self,
        sequences: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ) -> None:
        del is_prefill

        for sequence, token_id in zip(
            sequences,
            token_ids,
        ):
            sequence.append_token(token_id)

            reached_eos = (
                not sequence.sampling_params.ignore_eos
                and token_id == self.eos_token_id
            )

            reached_limit = (
                len(sequence.completion_token_ids)
                >= sequence.sampling_params.max_tokens
            )

            if reached_eos or reached_limit:
                sequence.status = SequenceStatus.FINISHED
                self.running.remove(sequence)


class CharacterTokenizer:
    """Tiny reversible tokenizer used only for testing."""

    eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        return [
            ord(character) + 3
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            chr(token_id - 3)
            for token_id in token_ids
            if token_id != self.eos_token_id
        )


class ScriptedModelRunner:
    """Returns A, B, then EOS for every request."""

    def __init__(self, eos_token_id: int) -> None:
        self.script = [
            ord("A") + 3,
            ord("B") + 3,
            eos_token_id,
        ]
        self.closed = False

    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        del is_prefill

        if self.closed:
            raise RuntimeError("runner is closed")

        return [
            self.script[
                min(
                    len(sequence.completion_token_ids),
                    len(self.script) - 1,
                )
            ]
            for sequence in sequences
        ]

    def close(self) -> None:
        self.closed = True


class CallStyleModelRunnerAdapter:
    """Adapt runner.call(method, ...) to run()/close()."""

    def __init__(self, runner: object) -> None:
        self.runner = runner

    def run(
        self,
        sequences: list[Sequence],
        is_prefill: bool,
    ) -> list[int]:
        return self.runner.call(
            "run",
            sequences,
            is_prefill,
        )

    def close(self) -> None:
        self.runner.call("exit")


def smoke_test() -> None:
    tokenizer = CharacterTokenizer()

    scheduler = SimpleScheduler(
        tokenizer.eos_token_id,
        max_num_sequences=2,
    )

    runner = ScriptedModelRunner(
        tokenizer.eos_token_id,
    )

    with LLMEngine(
        tokenizer,
        scheduler,
        runner,
        max_model_len=32,
    ) as engine:
        outputs = engine.generate(
            ["x", "y", "z"],
            SamplingParams(max_tokens=3),
        )

    expected = [
        {
            "text": "AB",
            "token_ids": [68, 69, 2],
        },
        {
            "text": "AB",
            "token_ids": [68, 69, 2],
        },
        {
            "text": "AB",
            "token_ids": [68, 69, 2],
        },
    ]

    assert outputs == expected
    print(outputs)

    
if __name__ == "__main__":
    smoke_test()