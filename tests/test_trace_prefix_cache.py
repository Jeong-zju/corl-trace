"""Full TRACE cached prefixes must not invoke the video decoder."""

import torch

from lerobot_policy_streaming_act.prefix_sequence import (
    DELTA_SIGNATURE_KEY,
    PATH_SIGNATURE_KEY,
    PREFIX_DELTA_SIGNATURE_KEY,
    PREFIX_PATH_SIGNATURE_KEY,
    PrefixSequenceDataset,
    prefix_image_key_from_camera_key,
)


def test_cached_full_trace_prefix_needs_no_video_decoder() -> None:
    camera_key = "observation.images.main"
    prefix_key = prefix_image_key_from_camera_key(camera_key)

    class _RowAccessor:
        def __getitem__(self, _idx):
            return type("Row", (), {"name": "toy-task"})()

    class _TaskTable:
        iloc = _RowAccessor()

    class _PrefixImageCacheReader:
        def has_key(self, key: str) -> bool:
            return key == camera_key

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            assert key == camera_key
            return torch.full((3, 2, 2), float(absolute_index), dtype=torch.float32)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            assert key == camera_key
            return torch.stack(
                [
                    torch.full((3, 2, 2), float(index), dtype=torch.float32)
                    for index in absolute_indices
                ],
                dim=0,
            )

    class _SignatureCacheReader:
        def has_key(self, key: str) -> bool:
            return key in {PATH_SIGNATURE_KEY, DELTA_SIGNATURE_KEY}

        def get(self, key: str, absolute_index: int) -> torch.Tensor:
            base = float(absolute_index)
            if key == PATH_SIGNATURE_KEY:
                return torch.tensor([base, base + 10.0], dtype=torch.float32)
            if key == DELTA_SIGNATURE_KEY:
                return torch.tensor([base, -base], dtype=torch.float32)
            raise KeyError(key)

        def get_many(self, key: str, absolute_indices: list[int]) -> torch.Tensor:
            return torch.stack([self.get(key, index) for index in absolute_indices], dim=0)

    class _BaseDataset:
        def __init__(self) -> None:
            self.hf_dataset = [
                {
                    "observation.state": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(0),
                    "timestamp": torch.tensor(0.0),
                    "task_index": torch.tensor(0),
                },
                {
                    "observation.state": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "episode_index": torch.tensor(0),
                    "index": torch.tensor(1),
                    "timestamp": torch.tensor(1.0),
                    "task_index": torch.tensor(0),
                },
            ]
            self.image_transforms = None
            self.features = {}
            self.delta_indices = None
            self._signature_cache_reader = _SignatureCacheReader()
            self.meta = type(
                "Meta",
                (),
                {
                    "camera_keys": (camera_key,),
                    "video_keys": (camera_key,),
                    "stats": {
                        "observation.state": {"mean": [0.0], "std": [1.0]},
                        PATH_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        DELTA_SIGNATURE_KEY: {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        camera_key: {"mean": [0.0], "std": [1.0]},
                    },
                    "episodes": [{"dataset_from_index": 0}],
                    "tasks": _TaskTable(),
                    "subtasks": None,
                },
            )()

        def __len__(self) -> int:
            return len(self.hf_dataset)

        def _ensure_hf_dataset_loaded(self) -> None:
            return None

        def _query_hf_dataset(self, query_indices):
            result = {}
            for key, indices in query_indices.items():
                if key == "observation.state":
                    result[key] = torch.stack(
                        [self.hf_dataset[index]["observation.state"] for index in indices],
                        dim=0,
                    )
                elif self._signature_cache_reader.has_key(key):
                    result[key] = self._signature_cache_reader.get_many(key, indices)
            return result

    dataset = PrefixSequenceDataset(
        _BaseDataset(),
        prefix_train_max_steps=2,
        prefix_frame_stride=1,
        prefix_pad_value=0.0,
        prefix_image_cache_reader=_PrefixImageCacheReader(),
    )

    item = dataset[1]

    assert torch.allclose(item[PATH_SIGNATURE_KEY], torch.tensor([1.0, 11.0]))
    assert torch.allclose(item[DELTA_SIGNATURE_KEY], torch.tensor([1.0, -1.0]))
    assert torch.allclose(
        item[PREFIX_PATH_SIGNATURE_KEY],
        torch.tensor([[0.0, 10.0], [1.0, 11.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        item[PREFIX_DELTA_SIGNATURE_KEY],
        torch.tensor([[0.0, -0.0], [1.0, -1.0]], dtype=torch.float32),
    )
    assert torch.allclose(item[prefix_key][1], torch.full((3, 2, 2), 1.0))
