# TODO: Post-MVP skeleton for patch EOT attack interface
from __future__ import annotations

from cyclops.attacks.base import Attack


class PatchEOTAttack(Attack):
    name = "patch_eot"
    requires_gradients = True

    def run(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("Patch EOT not implemented (post-MVP)")
