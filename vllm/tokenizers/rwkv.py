# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright BlinkDL and ChatRWKV contributors
#
# Adapted from BlinkDL/ChatRWKV tokenizer/rwkv_tokenizer.py.

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

from transformers import BatchEncoding

from .protocol import TokenizerLike

_VOCAB_FILE = Path(__file__).parent / "assets" / "rwkv_vocab_v20230424.txt"
_RWKV7_VOCAB_SIZE = 65536
_UNKNOWN_TOKEN_BYTES = "\ufffd".encode()


class _Trie:
    __slots__ = ("to", "token")

    def __init__(self) -> None:
        self.to: list[_Trie | None] = [None] * 256
        self.token = 0

    def add(self, key: bytes, value: int) -> None:
        node = self
        for byte in key:
            child = node.to[byte]
            if child is None:
                child = _Trie()
                node.to[byte] = child
            node = child
        node.token = value + 1


class RWKVTokenizer(TokenizerLike):
    """RWKV World tokenizer using the ChatRWKV byte-trie codec."""

    def __init__(
        self,
        vocab_file: str | Path = _VOCAB_FILE,
        name_or_path: str | Path | None = None,
        truncation_side: str = "left",
    ) -> None:
        if truncation_side not in {"left", "right"}:
            raise ValueError(
                f"truncation_side must be 'left' or 'right', got {truncation_side!r}"
            )

        idx2token: dict[int, bytes] = {}
        for line in Path(vocab_file).read_text(encoding="utf-8").splitlines():
            first_space = line.index(" ")
            last_space = line.rindex(" ")
            idx = int(line[:first_space])
            token = ast.literal_eval(line[first_space:last_space])
            token = token.encode("utf-8") if isinstance(token, str) else token
            if not isinstance(token, bytes):
                raise TypeError(f"RWKV vocab entry {idx} is not bytes or str")
            expected_length = int(line[last_space:])
            if len(token) != expected_length:
                raise ValueError(
                    f"RWKV vocab entry {idx} has {len(token)} bytes, "
                    f"expected {expected_length}"
                )
            idx2token[idx] = token

        self.token2idx = {token: idx for idx, token in idx2token.items()}
        self.idx2token = [_UNKNOWN_TOKEN_BYTES] * max(
            max(idx2token) + 1, _RWKV7_VOCAB_SIZE
        )
        self.idx2token[0] = b""
        for idx, token in idx2token.items():
            self.idx2token[idx] = token
        self._vocab = {
            self._token_to_vocab_key(token): idx
            for token, idx in self.token2idx.items()
        }

        self.name_or_path = str(name_or_path or vocab_file)
        self._max_chars_per_token = max(len(token) for token in self.idx2token)
        self._truncation_side = truncation_side

        self.root = _Trie()
        for token, idx in self.token2idx.items():
            self.root.add(token, value=idx)
        if any(child is None for child in self.root.to):
            raise ValueError("RWKV vocab must contain every single-byte token")

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> "RWKVTokenizer":
        del args, trust_remote_code, revision, download_dir
        vocab_file = kwargs.pop("vocab_file", _VOCAB_FILE)
        truncation_side = kwargs.pop("truncation_side", "left")
        return cls(
            vocab_file,
            name_or_path=path_or_repo_id,
            truncation_side=truncation_side,
        )

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def all_special_tokens(self) -> list[str]:
        return ["<|endoftext|>"]

    @property
    def all_special_ids(self) -> list[int]:
        return [0]

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        return len(self.idx2token)

    @property
    def max_token_id(self) -> int:
        return len(self.idx2token) - 1

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    def num_special_tokens_to_add(self) -> int:
        return 0

    def encode_bytes(self, source: bytes) -> list[int]:
        tokens: list[int] = []
        root_children = self.root.to
        offset = 0
        while offset < len(source):
            node = root_children[source[offset]]
            if node is None:
                raise ValueError(f"RWKV vocab does not contain byte {source[offset]}")

            end = offset + 1
            token = node.token
            cursor = end
            while cursor < len(source):
                child = node.to[source[cursor]]
                if child is None:
                    break
                node = child
                cursor += 1
                if node.token:
                    token = node.token
                    end = cursor

            if token == 0:
                raise ValueError("RWKV vocab trie matched no token")
            tokens.append(token - 1)
            offset = end
        return tokens

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        return b"".join(self.idx2token[token] for token in tokens)

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        del add_special_tokens
        tokens = self.encode_bytes(text.encode("utf-8"))
        if truncation and max_length is not None:
            if max_length < 0:
                raise ValueError(f"max_length must be non-negative, got {max_length}")
            if max_length == 0:
                return []
            if self.truncation_side == "left":
                tokens = tokens[-max_length:]
            else:
                tokens = tokens[:max_length]
        return tokens

    def decode(
        self,
        ids: Sequence[int] | int,
        skip_special_tokens: bool = False,
    ) -> str:
        token_ids = [ids] if isinstance(ids, int) else list(ids)
        if skip_special_tokens:
            token_ids = [
                token_id
                for token_id in token_ids
                if token_id not in self.all_special_ids
            ]
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        if text_pair is not None:
            raise NotImplementedError("text_pair is not supported for RWKVTokenizer.")

        if isinstance(text, list):
            input_ids = [
                self.encode(
                    value,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=add_special_tokens,
                )
                for value in text
            ]
            return BatchEncoding(
                {
                    "input_ids": input_ids,
                    "attention_mask": [[1] * len(ids) for ids in input_ids],
                }
            )

        input_ids = self.encode(
            text,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return BatchEncoding(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...

    @overload
    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        token_ids = (
            [token_id for token_id in ids if token_id not in self.all_special_ids]
            if skip_special_tokens
            else ids
        )
        return [
            self._token_to_vocab_key(self.idx2token[token_id]) for token_id in token_ids
        ]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        chunks: list[bytes] = []
        for token in tokens:
            try:
                chunks.append(token.encode("latin-1"))
            except UnicodeEncodeError:
                chunks.append(token.encode("utf-8"))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str | list[int]:
        del messages, tools, kwargs
        raise NotImplementedError("RWKVTokenizer does not define a chat template.")

    @staticmethod
    def _token_to_vocab_key(token: bytes) -> str:
        return token.decode("latin-1")

    def _convert_token_to_id(self, token: str) -> int:
        token_id = self._vocab.get(token)
        if token_id is not None:
            return token_id
        return self.token2idx[token.encode("utf-8")]
