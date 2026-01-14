# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# pyre-strict

from typing import Any, Generic, overload, TypeVar
from warnings import warn

T = TypeVar("T")


class DeprecatedAlias(Generic[T]):
    def __init__(self, new_name: str) -> None:
        self.new_name = new_name
        self.old_name = ""

    def __set_name__(self, owner: Any, name: str) -> None:
        self.old_name = name

    def _warn(self) -> None:
        warn(
            f"{self.old_name} is deprecated; use {self.new_name} instead",
            DeprecationWarning,
            stacklevel=3,
        )

    @overload
    # This case is required because the deprecated alias is a descriptor, i.e. it will
    # exist on the class, and therefore it will be called with obj=None.
    def __get__(self, obj: None, objtype: Any = ...) -> "DeprecatedAlias[T]": ...
    @overload
    def __get__(self, obj: object, objtype: Any = ...) -> T: ...

    def __get__(self, obj: object | None, objtype: Any = None) -> Any:
        if obj is None:
            return self
        self._warn()
        return getattr(obj, self.new_name)

    def __set__(self, obj: object, value: T) -> None:
        self._warn()
        setattr(obj, self.new_name, value)


class DeprecatedAttributesMixin:
    """
    Mixin providing backward-compatible deprecated attribute aliases.

    Add new deprecated attributes here as class variables using DeprecatedAlias.
    The actual attributes (e.g., monotone_t) must be defined by the class using this mixin.
    """

    MONOTONE_T = DeprecatedAlias[bool]("monotone_t")
    EARLY_STOPPING = DeprecatedAlias[bool]("early_stopping")
    EARLY_STOPPING_ESTIMATION_METHOD: DeprecatedAlias[Any] = DeprecatedAlias[Any](
        "early_stopping_estimation_method"
    )
    EARLY_STOPPING_TIMEOUT: DeprecatedAlias[int | None] = DeprecatedAlias[int | None](
        "early_stopping_timeout"
    )
    N_FOLDS: DeprecatedAlias[int] = DeprecatedAlias[int]("n_folds")
    NUM_ROUNDS: DeprecatedAlias[int] = DeprecatedAlias[int]("num_rounds")
    PATIENCE: DeprecatedAlias[int] = DeprecatedAlias[int]("patience")
