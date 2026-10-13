"""A version-checked correction to PyPOTS 1.5 BRITS validation loading.

Upstream ``BRITS.fit`` mistakenly constructs its validation DatasetForBRITS
from ``train_set``.  The subclass below changes that argument to ``val_set``;
the architecture, optimization, training loop, and checkpoint behavior remain
upstream implementations.  No installed package or global class is modified.

The copied fit method is from PyPOTS, revision
53b3eac34be9491ac3f28e65ee1993436e9318af,
pypots/classification/brits/model.py, created by Wenjie Du.
Its original BSD-3-Clause license is retained below.
"""

# Copyright (c) 2023-present, Wenjie Du
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import hashlib
import inspect
import io
import json
import textwrap
import tokenize
from importlib.metadata import version
from typing import Optional, Union

from pypots.classification.brits import BRITS as UpstreamBRITS
from pypots.imputation.brits.data import DatasetForBRITS
from torch.utils.data import DataLoader


_EXPECTED_VERSION = "1.5"
_EXPECTED_FIT_SHA256 = "1c4da2f00481758aaf707478004229c74e63523f27bcf2c34c2df1bb058667e9"


def _fit_fingerprint() -> str:
    """Hash source tokens, ignoring formatting and comments across Python versions."""
    source = textwrap.dedent(inspect.getsource(UpstreamBRITS.fit))
    ignored = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
    tokens = [
        # Numeric token IDs change between Python releases (OP: 54 -> 55).
        # Names remain stable for the tokens in this audited fit method.
        (tokenize.tok_name[token.type], token.string)
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in ignored
    ]
    serialized = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def check_brits_compatibility() -> None:
    """Fail clearly if the installed code is not the audited implementation."""
    installed_version = version("pypots")
    if installed_version != _EXPECTED_VERSION:
        raise RuntimeError(
            "The BRITS validation correction requires audited pypots==1.5; "
            f"found {installed_version}. Re-audit before changing the pin."
        )
    actual_fingerprint = _fit_fingerprint()
    if actual_fingerprint != _EXPECTED_FIT_SHA256:
        raise RuntimeError(
            "PyPOTS BRITS.fit differs from the audited implementation. "
            f"Expected {_EXPECTED_FIT_SHA256}, found {actual_fingerprint}. "
            "Re-audit its validation loading before applying this correction."
        )


class BRITSHoldoutValidation(UpstreamBRITS):
    """Official BRITS with its validation loader using the held-out split."""

    imts_compatibility = {
        "brits_heldout_validation_fix": True,
        "pypots_version": _EXPECTED_VERSION,
        "upstream_fit_sha256": _EXPECTED_FIT_SHA256,
        "upstream_fit_fingerprint_format": "token-names-v1",
        "change": "validation DatasetForBRITS(train_set) -> DatasetForBRITS(val_set)",
    }

    def fit(
        self,
        train_set: Union[dict, str],
        val_set: Optional[Union[dict, str]] = None,
        file_type: str = "hdf5",
    ) -> None:
        check_brits_compatibility()
        train_dataset = DatasetForBRITS(
            train_set,
            return_X_ori=False,
            return_y=True,
            file_type=file_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_dataloader = None
        if val_set is not None:
            val_dataset = DatasetForBRITS(
                val_set,  # The sole data/algorithm change from upstream fit().
                return_X_ori=False,
                return_y=True,
                file_type=file_type,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        self._train_model(train_dataloader, val_dataloader)
        self.model.load_state_dict(self.best_model_dict)
        self._auto_save_model_if_necessary(confirm_saving=self.model_saving_strategy == "best")
