"""Models and preprocessing for FMNIST, Tiny-ImageNet and IMDb (no data files needed)."""
import torch

from src.datasets.cifar import IMDB_MAXLEN, IMDB_PAD, IMDB_UNK, IMDB_VOCAB, Normalizer, _encode, _tokenize
from src.models import build_model


def test_lenet_fmnist_shape():
    assert build_model("lenet5", 10, "fmnist")(torch.zeros(2, 1, 28, 28)).shape == (2, 10)


def test_resnet_tinyimagenet_shape():
    assert build_model("resnet20", 200, "tinyimagenet")(torch.zeros(2, 3, 64, 64)).shape == (2, 200)


def test_textcnn_shape_and_vocab():
    m = build_model("textcnn", 2, "imdb")
    assert m.embedding.num_embeddings == IMDB_VOCAB + 2
    assert m(torch.zeros(2, IMDB_MAXLEN, dtype=torch.long)).shape == (2, 2)


def test_cifar_model_unchanged_without_dataset():
    assert build_model("lenet5", 10)(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


def test_normalizer_channels_and_passthrough():
    assert Normalizer("fmnist", "cpu")(torch.zeros(2, 1, 28, 28, dtype=torch.uint8)).shape == (2, 1, 28, 28)
    x = torch.ones(2, 5, dtype=torch.long)
    assert Normalizer("imdb", "cpu")(x) is x


def test_tokenize_and_encode():
    assert _tokenize("It's <br />GOOD!") == ["it's", "good"]
    ids = _encode(["a", "zzz"], {"a": 2})
    assert ids[:3] == [2, IMDB_UNK, IMDB_PAD] and len(ids) == IMDB_MAXLEN
    assert len(_encode(["a"] * (IMDB_MAXLEN + 5), {"a": 2})) == IMDB_MAXLEN
