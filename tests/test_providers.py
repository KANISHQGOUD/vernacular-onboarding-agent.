"""Contract tests for the three mocked providers.

These test the interfaces, not the mocks' cleverness. A real vendor dropped in
behind the same Protocol should pass the ones that describe the contract --
which is the point of writing them as interfaces at all.
"""

from __future__ import annotations

import inspect

import pytest

from app.providers.core_banking import (
    AccountOpeningFailure,
    CoreBankingProvider,
    MockCoreBanking,
)
from app.providers.kyc import (
    KycFailureReason,
    KycProvider,
    KycStatus,
    MockKycProvider,
    UnknownKycReferenceError,
)
from app.providers.payments import (
    FIRST_TRANSACTION_CEILING_PAISE,
    UPI_PER_TRANSACTION_LIMIT_PAISE,
    MockPaymentRail,
    PaymentFailure,
    PaymentRail,
    PaymentStatus,
    format_rupees,
)

# ---------------------------------------------------------------------------
# KYC -- the regulatory boundary expressed as a missing method.
# ---------------------------------------------------------------------------


def test_the_kyc_interface_offers_no_way_to_verify_anyone():
    """The bypass cannot be written because there is nothing to call.

    If someone later adds `verify()` or `set_status()` to this Protocol, this
    test fails and the reason is in front of them before they commit it.
    """
    methods = {name for name, _ in inspect.getmembers(KycProvider, inspect.isfunction)
               if not name.startswith("_")}
    assert methods == {"start_session", "get_status"}

    forbidden = ("verify", "approve", "set_status", "submit", "override", "decide")
    for name in methods:
        assert not any(word in name.lower() for word in forbidden)


def test_a_started_session_begins_pending_and_hands_over_a_url():
    provider = MockKycProvider()
    session = provider.start_session(applicant_reference="app-1", language="hi-IN")

    assert session.status is KycStatus.PENDING
    assert not session.is_settled
    assert session.reference.startswith("vckyc-")
    assert "hi-IN" in session.join_url


def test_polling_an_unknown_reference_raises():
    with pytest.raises(UnknownKycReferenceError):
        MockKycProvider().get_status("vckyc-never-issued")


def test_outcomes_are_scripted_rather_than_random():
    """A flaky test on the regulatory path teaches people to re-run it."""
    provider = MockKycProvider(
        outcomes=[
            (KycStatus.FAILED, KycFailureReason.NETWORK_DROPPED),
            (KycStatus.VERIFIED, None),
        ]
    )
    reference = provider.start_session("app-1", "hi-IN").reference

    first = provider.get_status(reference)
    assert first.status is KycStatus.FAILED
    assert first.failure_reason is KycFailureReason.NETWORK_DROPPED

    second = provider.get_status(reference)
    assert second.status is KycStatus.VERIFIED
    assert second.is_settled


def test_references_are_deterministic_across_identical_runs():
    a = MockKycProvider().start_session("app-1", "hi-IN")
    b = MockKycProvider().start_session("app-1", "hi-IN")
    assert a.reference == b.reference


# ---------------------------------------------------------------------------
# Core banking -- the constraint in the type signature.
# ---------------------------------------------------------------------------


def test_open_account_cannot_be_called_without_a_kyc_reference():
    """Not optional, and keyword-only. Two strings cannot be swapped by accident."""
    signature = inspect.signature(CoreBankingProvider.open_account)
    kyc_param = signature.parameters["kyc_reference"]

    assert kyc_param.default is inspect.Parameter.empty, "kyc_reference must be required"
    assert kyc_param.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        MockCoreBanking().open_account(full_name="A", mobile="9000000000")  # type: ignore[call-arg]


def test_an_empty_kyc_reference_is_rejected():
    result = MockCoreBanking().open_account(
        full_name="Asha Devi", mobile="9000000000", kyc_reference=""
    )
    assert not result.opened
    assert result.failure_reason is AccountOpeningFailure.KYC_REFERENCE_REJECTED


def test_a_fabricated_kyc_reference_is_rejected_downstream():
    """Even a well-formed reference must be one the KYC vendor actually issued."""
    bank = MockCoreBanking(valid_kyc_references={"vckyc-real"})

    good = bank.open_account(full_name="A", mobile="9000000001", kyc_reference="vckyc-real")
    assert good.opened

    bad = bank.open_account(full_name="B", mobile="9000000002", kyc_reference="vckyc-invented")
    assert not bad.opened
    assert bad.failure_reason is AccountOpeningFailure.KYC_REFERENCE_REJECTED


def test_existing_customers_are_found():
    bank = MockCoreBanking(existing_customers={"9876543210"})

    assert bank.find_customer("9876543210").has_active_account
    assert bank.find_customer("9876543210").customer_id
    assert not bank.find_customer("9000000000").has_active_account


def test_account_numbers_are_deterministic():
    a = MockCoreBanking().open_account(
        full_name="Asha Devi", mobile="9000000000", kyc_reference="vckyc-1"
    )
    b = MockCoreBanking().open_account(
        full_name="Asha Devi", mobile="9000000000", kyc_reference="vckyc-1"
    )
    assert a.account_number == b.account_number
    assert len(a.account_number) == 11


# ---------------------------------------------------------------------------
# Payments -- money is an integer.
# ---------------------------------------------------------------------------


def test_a_float_amount_is_rejected_loudly():
    """0.1 + 0.2 != 0.3. Currency in floats loses money in the fourth decimal."""
    rail = MockPaymentRail()

    with pytest.raises(TypeError, match="integer count of paise"):
        rail.initiate_deposit("0001", 100.50)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="integer count of paise"):
        rail.send_payment("0001", "a@upi", 100.50)  # type: ignore[arg-type]


def test_a_bool_is_not_an_amount():
    """bool is a subclass of int in Python, so True would otherwise pass as 1."""
    with pytest.raises(TypeError):
        MockPaymentRail().initiate_deposit("0001", True)  # type: ignore[arg-type]


def test_the_ledger_moves_in_whole_paise():
    rail = MockPaymentRail()
    assert rail.balance_paise("0001") == 0

    rail.initiate_deposit("0001", 50_000)   # Rs 500.00
    assert rail.balance_paise("0001") == 50_000

    result = rail.send_payment("0001", "shop@upi", 12_550)  # Rs 125.50
    assert result.succeeded
    assert rail.balance_paise("0001") == 37_450
    assert isinstance(rail.balance_paise("0001"), int)


def test_you_cannot_send_money_you_do_not_have():
    rail = MockPaymentRail()
    rail.initiate_deposit("0001", 10_000)

    result = rail.send_payment("0001", "shop@upi", 20_000)
    assert not result.succeeded
    assert result.failure_reason is PaymentFailure.INSUFFICIENT_BALANCE
    assert rail.balance_paise("0001") == 10_000, "a failed payment must not move the ledger"


@pytest.mark.parametrize("amount", [0, -1, -50_000])
def test_non_positive_amounts_are_refused(amount):
    result = MockPaymentRail().initiate_deposit("0001", amount)
    assert result.status is PaymentStatus.FAILED
    assert result.failure_reason is PaymentFailure.INVALID_AMOUNT


def test_the_upi_per_transaction_limit_is_enforced():
    rail = MockPaymentRail()
    rail.initiate_deposit("0001", UPI_PER_TRANSACTION_LIMIT_PAISE * 2)

    result = rail.send_payment("0001", "shop@upi", UPI_PER_TRANSACTION_LIMIT_PAISE + 1)
    assert result.failure_reason is PaymentFailure.LIMIT_EXCEEDED


def test_our_first_transaction_ceiling_is_lower_than_the_rails():
    """Two different limits, for two different reasons."""
    assert FIRST_TRANSACTION_CEILING_PAISE < UPI_PER_TRANSACTION_LIMIT_PAISE


@pytest.mark.parametrize(
    "paise,expected",
    [
        (0, "Rs 0.00"),
        (5, "Rs 0.05"),
        (100, "Rs 1.00"),
        (12_550, "Rs 125.50"),
        (10_000_000, "Rs 1,00,000.00".replace("1,00,000", "100,000")),
        (-2_550, "-Rs 25.50"),
    ],
)
def test_rupees_are_formatted_only_at_the_very_end(paise, expected):
    assert format_rupees(paise) == expected


def test_the_mocks_satisfy_their_protocols():
    """Structural check: a real vendor swaps in without touching call sites."""
    assert isinstance(MockKycProvider(), KycProvider)
    assert isinstance(MockCoreBanking(), CoreBankingProvider)
    assert isinstance(MockPaymentRail(), PaymentRail)
