from __future__ import annotations

import torch

from liquid_audio.data.dataloader import lfm2_collator
from liquid_audio.data.types import LFM2AudioRow
from liquid_audio.utils import LFMModality


def make_row(length: int, text_tokens: int) -> LFM2AudioRow:
    modality = torch.full((1, length), int(LFMModality.AUDIO_IN), dtype=torch.long)
    modality[:, :text_tokens] = int(LFMModality.TEXT)
    return LFM2AudioRow(
        text=torch.arange(text_tokens).unsqueeze(0),
        audio_in=torch.empty(128, 0),
        audio_in_lens=torch.empty(0, dtype=torch.long),
        audio_out=torch.empty(8, 0, dtype=torch.long),
        modality_flag=modality,
        supervision_mask=torch.zeros(1, length, dtype=torch.bool),
    )


def test_collator_dynamically_pads_to_longest_row() -> None:
    batch = lfm2_collator([make_row(4, 2), make_row(7, 3)])

    assert batch.modality_flag.shape == (2, 7)
    assert batch.supervision_mask.shape == (2, 7)
    assert batch.text.shape == (1, 8)
    assert torch.all(batch.modality_flag[0, 4:] == int(LFMModality.TEXT))
