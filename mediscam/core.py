"""MediSCAM: SPU, SLRCI, and CER for a frozen image classifier.

Input images must already be preprocessed for the supplied classifier.
The returned CAM has shape [1, 1, H, W] and values in [0, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _normalize(map_tensor: torch.Tensor) -> torch.Tensor:
    low = map_tensor.amin(dim=(-2, -1), keepdim=True)
    high = map_tensor.amax(dim=(-2, -1), keepdim=True)
    return (map_tensor - low) / (high - low).clamp_min(1e-8)


def _resize(maps: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(maps, size=size, mode="bilinear", align_corners=False)


class MediSCAM:
    """Generate a class activation map using the paper's three-stage method.

    For ResNet-18, use ``model.layer4`` and ``model.layer3``. For VGG-19,
    use ``model.features[35]`` and ``model.features[26]``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        guide_layer: torch.nn.Module,
        *,
        spu_strength: float = 0.4,
        interaction_topk: int = 32,
        contrast_strength: float = 0.3,
        interaction_strength: float = 0.05,
        interaction_cap: float = 0.05,
        score_batch_size: int = 64,
    ) -> None:
        self.model = model.eval()
        self.spu_strength = spu_strength
        self.interaction_topk = interaction_topk
        self.contrast_strength = contrast_strength
        self.interaction_strength = interaction_strength
        self.interaction_cap = interaction_cap
        self.score_batch_size = score_batch_size
        self._target_features: torch.Tensor | None = None
        self._guide_features: torch.Tensor | None = None
        self._hooks = [
            target_layer.register_forward_hook(self._capture_target),
            guide_layer.register_forward_hook(self._capture_guide),
        ]

    def _capture_target(self, _module, _inputs, output) -> None:
        self._target_features = output.detach().clone()

    def _capture_guide(self, _module, _inputs, output) -> None:
        self._guide_features = output.detach().clone()

    @staticmethod
    def _progressive(maps: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        height, width = size
        for scale in (16, 8, 4, 2, 1):
            maps = _resize(maps, (height // scale, width // scale))
        return maps

    @staticmethod
    def _smooth(maps: torch.Tensor) -> torch.Tensor:
        coords = torch.arange(-1, 2, device=maps.device, dtype=maps.dtype)
        kernel_1d = torch.exp(-coords.square() / 2)
        kernel = torch.outer(kernel_1d, kernel_1d)
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
        return F.conv2d(maps, kernel.expand(maps.shape[1], 1, 3, 3),
                        padding=1, groups=maps.shape[1])

    def _spu(self, activation: torch.Tensor, shallow: torch.Tensor,
             size: tuple[int, int]) -> torch.Tensor:
        guide = _normalize(F.relu(shallow).mean(dim=1, keepdim=True))
        guide = _normalize(_resize(guide, activation.shape[-2:]))
        guided = activation + self.spu_strength * activation * guide
        direct = self._smooth(_resize(activation, size))
        progressive = self._smooth(self._progressive(guided, size))
        return torch.tanh(torch.maximum(direct, progressive))

    def _channel_scores(self, image: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, maps.shape[1], self.score_batch_size):
            masks = maps[:, start:start + self.score_batch_size].permute(1, 0, 2, 3)
            logits = self.model(image * masks)
            chunks.append(F.softmax(logits, dim=1))
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _edge_response(maps: torch.Tensor) -> torch.Tensor:
        dx = torch.zeros_like(maps)
        dy = torch.zeros_like(maps)
        dx[..., :, 1:] = (maps[..., :, 1:] - maps[..., :, :-1]).abs()
        dy[..., 1:, :] = (maps[..., 1:, :] - maps[..., :-1, :]).abs()
        return (dx + dy).mean(dim=(0, 2, 3))

    def _slrci(self, maps: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # Keep the original per-channel rectification order.
        first = torch.zeros_like(maps[:, :1])
        for channel in range(maps.shape[1]):
            first = F.relu(first + scores[channel] * maps[:, channel:channel + 1])

        weights = F.relu(scores)
        count = min(self.interaction_topk, weights.numel())
        indices = weights.topk(count).indices
        selected_weights = weights.index_select(0, indices)
        selected_maps = maps.index_select(1, indices)
        if selected_weights.sum() < 1e-6:
            u = torch.full_like(selected_weights, 1.0 / count)
        else:
            u = selected_weights / (selected_weights.sum() + 1e-6)
        edge = self._edge_response(maps).index_select(0, indices)
        weighted_edge = selected_weights * edge
        if weighted_edge.sum() < 1e-6:
            v = u
        else:
            v = weighted_edge / (weighted_edge.sum() + 1e-6)
        projection_u = (u[None, :, None, None] * selected_maps).sum(dim=1, keepdim=True)
        projection_v = (v[None, :, None, None] * selected_maps).sum(dim=1, keepdim=True)
        interaction = _normalize(F.relu(projection_u * projection_v))

        first = _normalize(F.relu(first))
        agreement = ((first * interaction).sum() / (interaction.sum() + 1e-6)).clamp(0, 1)
        area = (interaction > 0.5).float().mean()
        if area < 0.03:
            area_gate = area / (0.03 + 1e-6)
        elif area > 0.40:
            area_gate = 0.40 / (area + 1e-6)
        else:
            area_gate = torch.ones_like(area)
        strength = (self.interaction_strength * agreement * area_gate).clamp(
            0, self.interaction_cap)
        return _normalize(F.relu(first + strength * first * interaction))

    def __call__(self, image: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """Explain one image; the competing class is the top non-target prediction."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("MediSCAM expects one image with shape [1, C, H, W].")
        with torch.no_grad():
            logits = self.model(image)
            activation = self._target_features
            shallow = self._guide_features
            if activation is None or shallow is None:
                raise RuntimeError("Both selected layers must run in the model forward pass.")
            probabilities = F.softmax(logits, dim=1)[0]
            target = int(probabilities.argmax()) if target_class is None else int(target_class)
            competitor_scores = probabilities.clone()
            competitor_scores[target] = -math.inf
            competitor = int(competitor_scores.argmax())

            maps = self._spu(activation, shallow, image.shape[-2:])
            channel_scores = self._channel_scores(image, maps)
            target_map = self._slrci(maps, channel_scores[:, target])
            competitor_map = self._slrci(maps, channel_scores[:, competitor])
            positive_evidence = _normalize(F.relu(target_map - competitor_map))
            return (target_map + self.contrast_strength *
                    F.relu(positive_evidence - target_map)).clamp(0, 1)

    def close(self) -> None:
        """Remove the feature hooks when the CAM generator is no longer needed."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
