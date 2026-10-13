"""The information contract: what a policy is allowed to read, enforced rather than documented.

PROPOSED EXTENSION. Every study in this laboratory states which quantities an agent may see; until now that was
kept true by convention and by review. A censored experiment makes convention too weak a guarantee. The whole
point of censoring is that total demand is unobservable, so a policy written for full feedback will reach for a
field that no longer exists for it - and if the field is still sitting in the dictionary it was handed, it will
silently keep its privileged access and the experiment will measure nothing.

This module makes that impossible rather than unlikely:
    InfoContract   names, for one policy, exactly which fields it may read
    InfoView       hands a policy its period record THROUGH the contract, and raises on anything else
    RESEARCHER_ONLY  fields no contract may ever grant, whatever it asks for

A policy never receives the record itself. It receives a view, and a view has no way to return a field the
contract does not name: not by indexing, not by attribute, not by iteration, not by copying the underlying
mapping out of it. The failure mode this prevents is silent, so the enforcement has to be loud.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Tuple


class InformationViolation(RuntimeError):
    """A policy tried to read a field its contract does not grant. Never caught inside a run: a violation means
    the experiment was not measuring what it claimed, so it stops."""


# Fields that belong to the researcher and to no agent, whatever a contract asks for. Granting one of these would
# make the agent an oracle; the oracle benchmark gets its advantage through its own declared contract instead, so
# that the privilege is visible in the record rather than hidden in a dictionary.
RESEARCHER_ONLY: Tuple[str, ...] = ("demand", "mu", "lost_sales", "regime", "true_gain", "oracle_order")


@dataclass(frozen=True)
class InfoContract:
    """What one policy may read each period, and why.

    `fields` is the whole of it. A field absent from this tuple cannot be reached through a view built from this
    contract, and a field in RESEARCHER_ONLY cannot be granted at all.
    """
    key: str
    name: str
    fields: Tuple[str, ...]
    note: str = ""
    may_acquire: bool = False          # whether this policy is allowed to buy information

    def __post_init__(self):
        bad = [f for f in self.fields if f in RESEARCHER_ONLY]
        if bad:
            raise InformationViolation(
                f"contract {self.key!r} asks for researcher-only field(s) {bad}: these name the truth the agent "
                f"is being tested on, and no contract may grant them")

    def grants(self, field: str) -> bool:
        return field in self.fields

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "name": self.name, "fields": list(self.fields), "note": self.note,
                "may_acquire": self.may_acquire}


class InfoView(Mapping):
    """A period record seen through a contract.

    Reading a granted field returns it. Reading anything else raises, and says which contract refused and what it
    would have allowed, because the usual cause is a policy written against a richer environment.
    """

    __slots__ = ("_granted", "_contract", "_reads")

    def __init__(self, record: Mapping[str, Any], contract: InfoContract):
        # The view keeps a COPY of the granted fields only. Hiding an attribute is not enforceable in Python -
        # a determined policy can reach any slot by name - so the view holds nothing it would be a violation to
        # reach. The refusals below make the mistake loud; this makes the circumvention empty.
        object.__setattr__(self, "_granted", {f: record[f] for f in contract.fields if f in record})
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(self, "_reads", [])

    @property
    def contract(self) -> InfoContract:
        return self._contract

    @property
    def reads(self) -> Tuple[str, ...]:
        """Fields actually read through this view, in order. Recorded so an experiment can report what a policy
        used rather than what it was entitled to use."""
        return tuple(self._reads)

    def __getitem__(self, field: str) -> Any:
        if not self._contract.grants(field):
            raise InformationViolation(
                f"policy {self._contract.key!r} read {field!r}, which its contract does not grant. "
                f"It may read: {', '.join(self._contract.fields) or '(nothing)'}."
                + (f" {field!r} is researcher-only." if field in RESEARCHER_ONLY else ""))
        if field not in self._granted:
            raise KeyError(f"{field!r} is granted by contract {self._contract.key!r} but absent from this record")
        self._reads.append(field)
        return self._granted[field]

    def __iter__(self) -> Iterator[str]:
        return iter(f for f in self._contract.fields if f in self._granted)

    def __len__(self) -> int:
        return len(self._granted)

    def __contains__(self, field: object) -> bool:
        return str(field) in self._granted

    def __getattr__(self, name: str) -> Any:
        # attribute access is not a side door: a policy reaching for .demand gets the same refusal as ["demand"]
        raise InformationViolation(
            f"policy {self._contract.key!r} tried to reach {name!r} as an attribute of its information view. "
            f"Fields are read by name, and only those its contract grants: "
            f"{', '.join(self._contract.fields) or '(nothing)'}.")

    def __repr__(self) -> str:
        return f"InfoView(contract={self._contract.key!r}, granted={list(self._contract.fields)})"
