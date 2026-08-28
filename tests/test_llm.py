"""The LLM layer, tested without spending quota.

Every test here runs offline against fakes. The free tier allows 5 requests a
minute, so a suite that called the API would be both slow and flaky -- and a
flaky test on the path that decides what a bank says to a customer is worse
than no test at all.

Live behaviour was verified separately, by hand, against the real API. What is
pinned here is the logic around the call: what gets offered to the model, what
is done with its answer, and what happens when it does not answer.
"""

from __future__ import annotations

import pytest
from google.genai import types

from app.flow.intents import classifiable_events
from app.flow.states import TERMINAL_STATES, Event, State
from app.llm.classifier import INTENT_GLOSS, GeminiIntentClassifier, build_prompt
from app.llm.phrasing import NATIVE_LANGUAGES, STATIC_ACTS, Phraser, build_instruction
from app.llm.speech_acts import SPEECH_ACTS, SpeechAct, SpeechActKind


class FakeClient:
    """Stands in for LlmClient. Records calls; returns scripted answers.

    `choose` is the intent name the model "returns"; None means the call
    failed, was rate-limited, or came back unusable -- the three cases the
    conversation treats identically.
    """

    def __init__(
        self,
        choose: str | None = None,
        text: str | None = None,
        field: str | None = None,
        value: str | None = None,
        language: str | None = None,
    ):
        self._answer = {
            "intent": choose,
            "field": field,
            "value": value,
            "language": language,
        }
        self._text = text
        self.extract_calls: list[types.Schema] = []
        self.say_calls: list[str] = []

    def extract(self, *, system_instruction, prompt, schema):
        self.extract_calls.append(schema)
        return dict(self._answer) if self._answer["intent"] is not None else None

    def say(self, *, system_instruction, instruction, max_output_tokens=512):
        self.say_calls.append(instruction)
        return self._text

    # -- Helpers so tests can inspect what the model was actually offered ---

    @property
    def offered_intents(self) -> set[str]:
        return set(self.extract_calls[0].properties["intent"].enum)

    @property
    def offered_fields(self) -> set[str]:
        return set(self.extract_calls[0].properties["field"].enum)


# ---------------------------------------------------------------------------
# Speech acts: the machine picks the subject, not the model.
# ---------------------------------------------------------------------------


def test_every_kind_has_a_spec():
    assert set(SPEECH_ACTS) == set(SpeechActKind)


@pytest.mark.parametrize("kind", sorted(SpeechActKind), ids=lambda v: str(v))
def test_every_act_can_speak_without_a_model(kind: SpeechActKind):
    """No act may depend on the API being up to produce a sentence."""
    text = SpeechAct(kind=kind).fallback_text()
    assert text.strip()
    assert not text.startswith("{")


def test_no_speech_act_sells_anything():
    """Same argument as the tool registry: closed by absence, not instruction.

    A model cannot upsell through a channel that has no speech act for it.
    """
    forbidden = ("offer", "upsell", "loan", "credit_card", "insurance", "recommend")
    for kind in SpeechActKind:
        assert not any(word in str(kind).lower() for word in forbidden), kind

    for spec in SPEECH_ACTS.values():
        blob = f"{spec.intent} {spec.fallback}".lower()
        for word in ("loan", "credit card", "insurance", "upgrade"):
            assert word not in blob, f"{spec.kind} mentions {word!r}"


def test_verbatim_fields_are_detected_when_dropped():
    act = SpeechAct(
        kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
        language="hi-IN",
        data={"account_number": "00012345678"},
    )
    assert act.missing_verbatim("Your account 00012345678 is open.") == ()
    assert act.missing_verbatim("Your account is open.") == ("account_number",)


def test_fallback_survives_missing_data():
    """An awkward sentence beats silence on a phone call."""
    text = SpeechAct(kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED).fallback_text()
    assert text.strip()


# ---------------------------------------------------------------------------
# Phrasing: the quota-shaped decisions.
# ---------------------------------------------------------------------------


def test_english_never_calls_the_model():
    """The fallbacks are already correct English. Paying to rewrite them
    would spend a fifth of the per-minute budget to change nothing."""
    client = FakeClient(text="should not be used")
    phraser = Phraser(client=client)

    for language in sorted(NATIVE_LANGUAGES):
        phraser.say(SpeechAct(kind=SpeechActKind.ASK_LANGUAGE, language=language))

    assert client.say_calls == []
    assert phraser.stats()["calls_made"] == 0


def test_the_same_act_is_bought_once():
    client = FakeClient(text="क्या मैं आगे बढ़ूँ?")
    phraser = Phraser(client=client)
    act = SpeechAct(kind=SpeechActKind.EXPLAIN_CONSENT, language="hi-IN")

    first = phraser.say(act)
    for _ in range(9):
        assert phraser.say(act) == first

    assert len(client.say_calls) == 1
    assert phraser.stats()["cache_hits"] == 9


def test_prewarm_pays_before_anyone_is_waiting():
    client = FakeClient(text="नमस्ते")
    phraser = Phraser(client=client)

    made = phraser.prewarm("hi-IN")
    assert made == len(STATIC_ACTS)

    # Those acts are now free for the rest of the conversation.
    before = phraser.calls_made
    for kind in STATIC_ACTS:
        phraser.say(SpeechAct(kind=kind, language="hi-IN"))
    assert phraser.calls_made == before


def test_prewarm_is_a_no_op_for_english():
    phraser = Phraser(client=FakeClient(text="x"))
    assert phraser.prewarm("en-IN") == 0


def test_a_rate_limit_still_produces_a_sentence():
    """None from the client means 429, outage, or empty reply. All three must
    still speak -- an agent that goes quiet has hung up on somebody."""
    phraser = Phraser(client=FakeClient(text=None))
    said = phraser.say(SpeechAct(kind=SpeechActKind.EXPLAIN_CONSENT, language="hi-IN"))

    assert said.strip()
    assert said == SpeechAct(kind=SpeechActKind.EXPLAIN_CONSENT).fallback_text()
    assert phraser.stats()["fallbacks_used"] == 1


def test_a_dropped_account_number_discards_the_translation():
    """Misreporting an account number is worse than failing to translate."""
    client = FakeClient(text="आपका खाता खुल गया है।")  # number missing
    phraser = Phraser(client=client)

    act = SpeechAct(
        kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
        language="hi-IN",
        data={"account_number": "00012345678"},
    )
    said = phraser.say(act)

    assert "00012345678" in said
    assert phraser.stats()["fallbacks_used"] == 1


def test_a_rounded_amount_is_rejected():
    client = FakeClient(text="लगभग सवा सौ रुपये का भुगतान हो गया।")
    phraser = Phraser(client=client)

    act = SpeechAct(
        kind=SpeechActKind.CONFIRM_COMPLETED,
        language="hi-IN",
        data={"amount_text": "Rs 125.50"},
    )
    assert "Rs 125.50" in phraser.say(act)


def test_a_good_translation_is_kept():
    client = FakeClient(text="आपका खाता 00012345678 खुल गया है।")
    phraser = Phraser(client=client)
    said = phraser.say(
        SpeechAct(
            kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
            language="hi-IN",
            data={"account_number": "00012345678"},
        )
    )
    assert said == "आपका खाता 00012345678 खुल गया है।"
    assert phraser.stats()["fallbacks_used"] == 0


def test_no_client_at_all_still_speaks():
    phraser = Phraser(client=None)
    assert phraser.say(SpeechAct(kind=SpeechActKind.ASK_LANGUAGE, language="ta-IN")).strip()


def test_the_instruction_names_the_values_that_must_survive():
    instruction = build_instruction(
        SpeechAct(
            kind=SpeechActKind.ANNOUNCE_ACCOUNT_OPENED,
            language="hi-IN",
            data={"account_number": "00012345678"},
        )
    )
    assert "hi-IN" in instruction
    assert instruction.count("00012345678") >= 2  # listed, then required


# ---------------------------------------------------------------------------
# Classifier: still fenced, whichever model is behind it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", sorted(TERMINAL_STATES), ids=lambda v: str(v))
def test_terminal_states_spend_no_quota(state: State):
    """A finished session accepts nothing. Do not pay to be told so."""
    client = FakeClient(choose="CONSENT_GIVEN")
    assert GeminiIntentClassifier(client).classify("hello", state) is None
    assert client.extract_calls == []


def test_silence_spends_no_quota():
    client = FakeClient(choose="CONSENT_GIVEN")
    assert GeminiIntentClassifier(client).classify("   ", State.CONSENT) is None
    assert client.extract_calls == []


def test_the_offered_options_never_exceed_the_map():
    """What the model may say is narrowed before it is ever asked."""
    client = FakeClient(choose="CONSENT_GIVEN")
    GeminiIntentClassifier(client).classify("haan ji", State.CONSENT)

    offered = {Event[name] for name in client.offered_intents}
    assert offered == classifiable_events(State.CONSENT)


def test_a_valid_answer_becomes_an_event():
    classifier = GeminiIntentClassifier(FakeClient(choose="CONSENT_GIVEN"))
    assert classifier.classify("haan ji", State.CONSENT) is Event.CONSENT_GIVEN


def test_an_unparseable_answer_is_a_reprompt_not_a_crash():
    classifier = GeminiIntentClassifier(FakeClient(choose="NOT_AN_EVENT"))
    assert classifier.classify("haan ji", State.CONSENT) is None


def test_a_model_failure_is_a_reprompt_not_a_crash():
    classifier = GeminiIntentClassifier(FakeClient(choose=None))
    assert classifier.classify("haan ji", State.CONSENT) is None


def test_every_offerable_event_has_a_gloss():
    """The model matches sentences against these descriptions, so an event
    with no gloss is one it will guess at."""
    offerable = set()
    for state in State:
        offerable |= classifiable_events(state)
    missing = offerable - set(INTENT_GLOSS)
    assert not missing, f"no description for: {missing}"


def test_the_prompt_carries_the_menu_and_the_words():
    offered = sorted(classifiable_events(State.CONSENT), key=lambda e: e.name)
    prompt = build_prompt("haan ji theek hai", offered)

    assert "haan ji theek hai" in prompt
    for event in offered:
        assert event.name in prompt


def test_an_api_failure_raises_rather_than_looking_like_a_mumble():
    """`None` used to mean both "understood nothing" and "could not reach the
    thing that understands". Those need different answers, so only one of them
    is still None."""
    from google.genai import errors as genai_errors
    from google.genai import types

    from app.llm.client import LlmClient, ModelUnavailableError

    class Down:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                raise genai_errors.ServerError(503, {"error": {"message": "high demand"}})

    client = LlmClient(api_key="x")
    client._client = Down()

    with pytest.raises(ModelUnavailableError):
        client.extract(
            system_instruction="system",
            prompt="hindi",
            schema=types.Schema(type=types.Type.OBJECT),
        )
