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


# ----------------------------------------------------------------- Vietnamese
def test_vietnamese_capitalises_diacritic_letters():
    assert normalize("được rồi. đi thôi", language="vi") == "Được rồi. Đi thôi."
    assert normalize("ừ ,ổn rồi", language="vi", remove_fillers=False) == "Ừ, ổn rồi."


def test_vietnamese_missing_space_after_punctuation():
    assert normalize("xin chào,mọi người.hôm nay họp", language="vi") == "Xin chào, mọi người. Hôm nay họp."


def test_vietnamese_fillers_keep_particles():
    assert normalize("ờ, anh đi à?", language="vi") == "Anh đi à?"
    assert normalize("ừm vâng ạ", language="vi") == "Vâng ạ."


def test_vietnamese_decimal_comma_and_lone_i():
    assert normalize("tăng 3,5 phần trăm", language="vi") == "Tăng 3,5 phần trăm."
    assert normalize("chữ i thôi", language="vi") == "Chữ i thôi."


def test_nfc_normalisation():
    decomposed = "đường"  # "đường" with a combining grave accent
    assert normalize(decomposed, language="vi") == "Đường."


def test_vietnamese_hallucinations():
    assert is_hallucination("Hãy subscribe cho kênh Ghiền Mì Gõ để không bỏ lỡ video")
    assert is_hallucination(" Cảm ơn các bạn đã theo dõi. ")
    assert not is_hallucination("Cảm ơn anh Nam đã trình bày.")
