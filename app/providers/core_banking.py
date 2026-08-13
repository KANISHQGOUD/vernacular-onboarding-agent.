"""Core banking -- the system of record that actually holds accounts.

Two jobs, at two different points in the flow:

    find_customer()   during ELIGIBILITY, to answer "do they already bank here?"
    open_account()    during ACCOUNT_CREATE, to actually create one

The signature of `open_account` is the part worth studying. It takes
`kyc_reference` as a required argument, so it is not possible to phrase a call
that opens an account without one. That constraint already exists in the state
machine -- ACCOUNT_CREATE is reachable only via KYC_VERIFIED -- but the state
machine protects the path, not the function. Anyone who imported this module
and called it directly would be outside the machine entirely.

Requiring the reference in the type signature means the guarantee survives that
mistake. It is a small thing to type and it removes a whole category of "we
called it from the wrong place" incident.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class AccountOpeningFailure(StrEnum):
    KYC_REFERENCE_REJECTED = "KYC_REFERENCE_REJECTED"
    DUPLICATE_CUSTOMER = "DUPLICATE_CUSTOMER"
    BRANCH_SYSTEM_DOWN = "BRANCH_SYSTEM_DOWN"
    NAME_SCREENING_HIT = "NAME_SCREENING_HIT"


@dataclass(frozen=True, slots=True)
class CustomerLookup:
    """Answer to 'is this person already a customer?'

    Feeds the NO_EXISTING_ACCOUNT rule in eligibility.py. Note the direction of
    travel: the provider supplies a fact, eligibility applies a rule to it. The
    provider does not decide eligibility, and the rule does not make network
    calls -- which is what keeps the ruleset pure and replayable.
    """

    has_active_account: bool
    customer_id: str | None = None


@dataclass(frozen=True, slots=True)
class OpenedAccount:
    opened: bool
    account_number: str | None = None
    ifsc: str | None = None
    failure_reason: AccountOpeningFailure | None = None


@runtime_checkable
class CoreBankingProvider(Protocol):
    def find_customer(self, mobile: str, pan: str | None = None) -> CustomerLookup: ...

    def open_account(
        self,
        *,
        full_name: str,
        mobile: str,
        kyc_reference: str,
    ) -> OpenedAccount: ...


@dataclass
class MockCoreBanking:
    """Stand-in for the bank's system of record.

    `existing_customers` seeds mobiles that already bank here, so the
    ALREADY_A_CUSTOMER decline path can be driven from a real lookup rather
    than by hand-setting a boolean.
    """

    existing_customers: set[str] = field(default_factory=set)
    valid_kyc_references: set[str] | None = None
    fail_next_opening_with: AccountOpeningFailure | None = None

    _opened: dict[str, OpenedAccount] = field(default_factory=dict)

    def find_customer(self, mobile: str, pan: str | None = None) -> CustomerLookup:
        if mobile in self.existing_customers:
            digest = hashlib.sha256(mobile.encode()).hexdigest()[:10].upper()
            return CustomerLookup(has_active_account=True, customer_id=f"CIF{digest}")
        return CustomerLookup(has_active_account=False)

    def open_account(
        self,
        *,
        full_name: str,
        mobile: str,
        kyc_reference: str,
    ) -> OpenedAccount:
        # Keyword-only arguments, so no caller can pass these positionally and
        # silently swap `mobile` and `kyc_reference` -- two strings that would
        # otherwise substitute for each other without complaint.
        if not kyc_reference:
            return OpenedAccount(
                opened=False, failure_reason=AccountOpeningFailure.KYC_REFERENCE_REJECTED
            )

        # A real core banking system validates the reference against the KYC
        # vendor. Leaving this configurable lets a test prove that a fabricated
        # reference is rejected downstream, not merely upstream.
        if self.valid_kyc_references is not None and kyc_reference not in self.valid_kyc_references:
            return OpenedAccount(
                opened=False, failure_reason=AccountOpeningFailure.KYC_REFERENCE_REJECTED
            )

        if self.fail_next_opening_with is not None:
            failure = self.fail_next_opening_with
            self.fail_next_opening_with = None
            return OpenedAccount(opened=False, failure_reason=failure)

        if mobile in self.existing_customers:
            return OpenedAccount(
                opened=False, failure_reason=AccountOpeningFailure.DUPLICATE_CUSTOMER
            )

        # Deterministic account number, so a replayed run produces an identical
        # audit trail rather than a fresh random one.
        digest = hashlib.sha256(f"{full_name}:{mobile}".encode()).hexdigest()
        account_number = str(int(digest[:12], 16))[:11].rjust(11, "0")

        account = OpenedAccount(opened=True, account_number=account_number, ifsc="SBIN0000001")
        self._opened[account_number] = account
        self.existing_customers.add(mobile)
        return account
