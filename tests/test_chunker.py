from wiki_chat.chunker import chunk_doc, chunk_id_for


def test_title_is_prefixed_in_chunk_text():
    chunks = chunk_doc("牛顿", "牛顿第二定律表明 F=ma。", size=50, overlap=10)
    assert len(chunks) >= 1
    assert chunks[0].startswith("标题：牛顿\n\n")


def test_short_text_returns_single_chunk():
    chunks = chunk_doc("标题A", "短文本", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "标题：标题A\n\n短文本"


def test_overlap_present_between_consecutive_chunks():
    text = "一二三四五六七八九十" * 100  # 1000 字符
    chunks = chunk_doc("T", text, size=50, overlap=15)
    assert len(chunks) >= 3
    tail = chunks[0][-15:]
    assert any(tail[:8] in c for c in chunks[1:])


def test_chunks_break_at_newline_when_close_to_end():
    text = "ABCDE\n" + "X" * 50 + "\nFGHIJ"
    chunks = chunk_doc("T", text, size=30, overlap=5)
    assert chunks[0].rstrip().endswith("ABCDE")


def test_empty_text_returns_empty_list():
    assert chunk_doc("T", "", size=100, overlap=10) == []


def test_very_long_text_produces_many_chunks():
    text = "X" * 5000
    chunks = chunk_doc("T", text, size=100, overlap=20)
    # stride ≈ (100-20)*1.5 = 120 字符；5000/120 ≈ 42
    assert 30 <= len(chunks) <= 60


def test_chunk_id_format():
    assert chunk_id_for("13", 0) == "13#0"
    assert chunk_id_for("13", 7) == "13#7"
