from meetingtool.textnorm import is_hallucination, normalize


def test_adds_period_and_capitalises():
    assert normalize("so we should ship it") == "So we should ship it."


def test_spacing_and_duplicates():
    assert normalize("well ,, i think so .yes") == "Well, I think so. Yes."


def test_ellipsis_and_fillers():
    assert normalize("um, we could... uh maybe wait...") == "We could, maybe wait."


def test_keeps_decimals_and_questions():
    assert normalize("version 3.5 is out? great") == "Version 3.5 is out? Great."


def test_i_contractions():
    assert normalize("i'm sure i'll do it") == "I'm sure I'll do it."


def test_hallucination_list():
    assert is_hallucination(" Thank you. ")
    assert not is_hallucination("Thank you, Anna.")
