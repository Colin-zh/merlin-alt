"""xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems."""

from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from ..base_model import BaseModel
from ..fm.fm_model import NumEmbedding, CatEmbedding, CatLinear


class CIN(nn.Module):
	"""Compressed Interaction Network for explicit high-order feature interactions."""

	def __init__(self, field_dims: int, embed_dim: int, layer_sizes: List[int], split_half: bool = True):
		super().__init__()
		self.field_dims = field_dims
		self.embed_dim = embed_dim
		self.layer_sizes = layer_sizes
		self.split_half = split_half

		self.conv_layers = nn.ModuleList()
		prev_field_dim = field_dims
		for size in layer_sizes:
			self.conv_layers.append(nn.Conv1d(in_channels=prev_field_dim * field_dims, out_channels=size, kernel_size=1))
			prev_field_dim = size // 2 if split_half else size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, field_num, embed_dim)
		x0 = x
		xs = []
		h = x
		for i, conv in enumerate(self.conv_layers):
			# outer product between x0 and h along field dimension
			# (B, f0, 1, d) * (B, 1, fi, d) -> (B, f0, fi, d)
			xz = torch.einsum("bhd,bmd->bhmd", x0, h)  # (B, f0, fi, d)
			xz = xz.reshape(xz.size(0), xz.size(1) * xz.size(2), self.embed_dim)  # (B, f0*fi, d)
			xz = xz.permute(0, 2, 1)  # (B, d, f0*fi)
			z = conv(xz)  # (B, size, f0*fi)
			z = F.relu(z)
			z = z.permute(0, 2, 1)  # (B, f_new, size)

			if self.split_half and i != len(self.conv_layers) - 1:
				next_h, part = torch.split(z, z.size(1) // 2, dim=1)
			else:
				next_h, part = z, z
			xs.append(part)
			h = next_h

		# Sum over embedding dimension for each layer's part
		result = torch.cat(xs, dim=1)  # (B, sum_fields, embed_dim)
		result = torch.sum(result, dim=2)  # (B, sum_fields)
		return result


class MLP(nn.Module):
	def __init__(self, d_in: int, d_layers: List[int], dropout: float = 0.0, d_out: int = 1):
		super().__init__()
		layers = []
		for d in d_layers:
			layers.append(nn.Linear(d_in, d))
			layers.append(nn.BatchNorm1d(d))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
			d_in = d
		layers.append(nn.Linear(d_in, d_out))
		self.mlp = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.mlp(x)


class XDeepFMBackbone(nn.Module):
	def __init__(self, config, **kwargs):
		super().__init__()
		self.hparams = config

		d_embed = self.hparams.input_embed_dim
		categories = self.hparams.categorical_cardinality or []
		d_numerical = self.hparams.continuous_dim or 0
		n_classes = self.hparams.output_dim

		self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
		self.cat_linear = CatLinear(categories, n_classes) if categories else None

		self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
		self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

		self.field_num = (len(categories) if categories else 0) + (d_numerical if d_numerical else 0)

		cin_layers = [int(d) for d in self.hparams.cin_layer_sizes.split("-")]
		self.cin = CIN(self.field_num, d_embed, cin_layers, split_half=self.hparams.cin_split_half)
		self.cin_linear = nn.Linear(sum(cin_layers), n_classes)

		deep_layers = [int(d) for d in self.hparams.deep_layers.split("-")]
		self.deep = MLP(d_in=self.field_num * d_embed, d_layers=deep_layers, dropout=self.hparams.deep_dropout, d_out=n_classes)

		self.output_dim = n_classes

	def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
		x_cat, x_num = x.get("categorical"), x.get("numerical") or x.get("continuous")

		linear_out = 0.0
		if self.num_linear is not None and x_num is not None:
			linear_out = linear_out + self.num_linear(x_num)
		if self.cat_linear is not None and x_cat is not None:
			linear_out = linear_out + self.cat_linear(x_cat)

		embeds = []
		if self.num_embedding is not None and x_num is not None:
			embeds.append(self.num_embedding(x_num[..., None]))
		if self.cat_embedding is not None and x_cat is not None:
			embeds.append(self.cat_embedding(x_cat))

		if not embeds:
			raise ValueError("XDeepFMBackbone expects numerical or categorical inputs.")

		x_embed = torch.cat(embeds, dim=1)  # (B, field_num, d_embed)

		# CIN part
		cin_out = self.cin(x_embed)  # (B, sum(cin_layers))
		cin_out = self.cin_linear(cin_out)

		# Deep part
		deep_out = self.deep(x_embed.reshape(x_embed.size(0), -1))

		return linear_out + cin_out + deep_out


class XDeepFMModel(BaseModel):
	def __init__(self, config, **kwargs):
		super().__init__(config, **kwargs)

	@property
	def backbone(self):
		return self._backbone

	@property
	def embedding_layer(self):
		return self._embedding_layer

	@property
	def head(self):
		return self._head

	def _build_network(self):
		self._embedding_layer = nn.Identity()
		self._backbone = XDeepFMBackbone(self.hparams)
		setattr(self.backbone, "output_dim", self.hparams.output_dim)
		self._head = nn.Identity()

	def _build_model(self):
		return self._build_network()

	def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, Any]:
		x = self.embed_input(x)
		x = self.compute_backbone(x)
		return self.compute_head(x)

	def extract_embedding(self):
		raise ValueError("Extracting Embeddings is not supported by XDeepFMModel.")