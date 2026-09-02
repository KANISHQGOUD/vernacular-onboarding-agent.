"""Payment rail -- the two steps the whole product is actually measured on.

FUNDING gets money into the account. FIRST_TRANSACTION gets money out of it, to
somebody. An account that reaches the first and not the second is the dormant
account this product exists to prevent, which is why they are separate states
with separate provider calls rather than one "activate" step.

MONEY IS AN INTEGER HERE, ALWAYS.

Every amount in this module is in paise -- hundredths of a rupee -- and typed
as `int`. Never a float, never a string, never rupees.

The reason is that binary floating point cannot represent most decimal
fractions. 0.1 + 0.2 is 0.30000000000000004, and ~1.1 lakh is not exactly
representable either. Sum enough float rupees and the total drifts from what
the ledger says. The drift is tiny and it is also unacceptable: a bank that
cannot reconcile to the paisa has a regulatory problem, not a rounding
preference.

So the rule is that money crosses every boundary in this system as an integer
count of the smallest unit, and only becomes "Rs 1,250.00" at the last moment,
for a human to read or hear.
"""

from __future__ import annotations

import re
from decimal import Decimal

from app.speech.wer import DIGIT_MAP

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

PAISE_PER_RUPEE = 100

# UPI per-transaction ceiling for ordinary payments: Rs 1,00,000.
UPI_PER_TRANSACTION_LIMIT_PAISE = 100_000 * PAISE_PER_RUPEE

# A first transaction should be small. Someone learning to use their account
# for the first time should not be walked into a large transfer by a voice they
# have known for four minutes.
FIRST_TRANSACTION_CEILING_PAISE = 500 * PAISE_PER_RUPEE


class PaymentStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


class PaymentFailure(StrEnum):
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    INVALID_AMOUNT = "INVALID_AMOUNT"
    LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
    BENEFICIARY_INVALID = "BENEFICIARY_INVALID"
    RAIL_UNAVAILABLE = "RAIL_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PaymentResult:
    reference: str | None
    status: PaymentStatus
    amount_paise: int
    failure_reason: PaymentFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is PaymentStatus.SUCCESS


# A number in the middle of a sentence: "500", "Rs 500", "500 rupees",
# "1,000", "12.50". The decimal part is capped at two digits because paise
# stop there; a third would be the recogniser inventing precision.
_AMOUNT = re.compile(r"(\d+(?:\.\d{1,2})?)")


def paise_from_text(text: str) -> int | None:
    """A spoken rupee amount as integer paise, or None if there is no number.

    Money is integer paise everywhere in this system, and this is the seam
    where a sentence becomes one -- so it is the one place that has to resist
    floats. `Decimal` rather than `float(...) * 100`, because 12.55 is not
    representable in binary and `int(12.55 * 100)` is 1254.

    Devanagari and the other Indic digits need no folding here, which is
    worth stating because the obvious defensive `translate(DIGIT_MAP)` looks
    load-bearing and is not: Python's `\\d` and `Decimal` both accept any
    Unicode decimal digit, so `१०००` parses as a thousand unaided. Mutation
    testing is what established that -- removing a fold changed no behaviour
    at all. `fields.as_date` genuinely does need one, because
    `date.fromisoformat` is stricter than `Decimal`.

    Thousands separators are stripped before the decimal point is read, so
    "1,000" is a thousand rupees rather than one rupee.
    """
    if not isinstance(text, str):
        return None

    folded = text.replace(",", "")
    found = _AMOUNT.search(folded)
    if not found:
        return None

    rupees = Decimal(found[1])
    if rupees <= 0:
        return None
    return int(rupees * PAISE_PER_RUPEE)


def format_rupees(amount_paise: int) -> str:
    """Integer paise to a human string, at the very last moment.

    This is the only place the two-decimal shape appears, and it still never
    does float arithmetic -- divmod on integers, then formatting.
    """
    rupees, paise = divmod(abs(amount_paise), PAISE_PER_RUPEE)
    sign = "-" if amount_paise < 0 else ""
    return f"{sign}Rs {rupees:,}.{paise:02d}"


@runtime_checkable
class PaymentRail(Protocol):
    def initiate_deposit(self, account_number: str, amount_paise: int) -> PaymentResult: ...

    def send_payment(
        self, from_account: str, to_vpa: str, amount_paise: int
    ) -> PaymentResult: ...

    def balance_paise(self, account_number: str) -> int: ...


@dataclass
class MockPaymentRail:
    """Stand-in for UPI/NEFT. Keeps a paise ledger so balances actually move."""

    fail_next_with: PaymentFailure | None = None
    known_vpas: set[str] | None = None

    _balances: dict[str, int] = field(default_factory=dict)
    _sequence: int = 0

    def balance_paise(self, account_number: str) -> int:
        return self._balances.get(account_number, 0)

    def initiate_deposit(self, account_number: str, amount_paise: int) -> PaymentResult:
        if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
            # A float that reached this far is a bug upstream, and swallowing it
            # would be the expensive kind of quiet.
            raise TypeError(
                f"amount must be an integer count of paise, got {type(amount_paise).__name__}"
            )

        if amount_paise <= 0:
            return self._failed(amount_paise, PaymentFailure.INVALID_AMOUNT)

        if (failure := self._take_scheduled_failure()) is not None:
            return self._failed(amount_paise, failure)

        self._balances[account_number] = self.balance_paise(account_number) + amount_paise
        return self._succeeded(amount_paise, f"dep-{account_number}")

    def send_payment(
        self, from_account: str, to_vpa: str, amount_paise: int
    ) -> PaymentResult:
        if not isinstance(amount_paise, int) or isinstance(amount_paise, bool):
            raise TypeError(
                f"amount must be an integer count of paise, got {type(amount_paise).__name__}"
            )

        if amount_paise <= 0:
            return self._failed(amount_paise, PaymentFailure.INVALID_AMOUNT)

        if amount_paise > UPI_PER_TRANSACTION_LIMIT_PAISE:
            return self._failed(amount_paise, PaymentFailure.LIMIT_EXCEEDED)

        if self.known_vpas is not None and to_vpa not in self.known_vpas:
            return self._failed(amount_paise, PaymentFailure.BENEFICIARY_INVALID)

        if (failure := self._take_scheduled_failure()) is not None:
            return self._failed(amount_paise, failure)

        if self.balance_paise(from_account) < amount_paise:
            return self._failed(amount_paise, PaymentFailure.INSUFFICIENT_BALANCE)

        self._balances[from_account] -= amount_paise
        return self._succeeded(amount_paise, f"upi-{to_vpa}")

    # -- internals ----------------------------------------------------------

    def _take_scheduled_failure(self) -> PaymentFailure | None:
        failure, self.fail_next_with = self.fail_next_with, None
        return failure

    def _succeeded(self, amount_paise: int, seed: str) -> PaymentResult:
        self._sequence += 1
        digest = hashlib.sha256(f"{seed}:{self._sequence}".encode()).hexdigest()[:10]
        return PaymentResult(
            reference=digest, status=PaymentStatus.SUCCESS, amount_paise=amount_paise
        )

    @staticmethod
    def _failed(amount_paise: int, reason: PaymentFailure) -> PaymentResult:
        return PaymentResult(
            reference=None,
            status=PaymentStatus.FAILED,
            amount_paise=amount_paise,
            failure_reason=reason,
        )
