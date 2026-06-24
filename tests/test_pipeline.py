from pathlib import Path
from unittest.mock import patch

from campus_rag.config import AppConfig
from campus_rag.document_loader import load_document
from campus_rag.pipeline import CampusRagPipeline


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project_root=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        finetune_dir=tmp_path / "finetune",
        max_chars=500,
        overlap_chars=50,
        recall_top_k=5,
        min_score=0.05,
        similarity_weight=0.65,
        keyword_weight=0.25,
        completeness_weight=0.10,
        embedding_backend="tfidf",
        embedding_model_name="",
        embedding_device="auto",
        llm_backend="extractive",
        llm_model_name="",
        lora_adapter=None,
        max_new_tokens=128,
        temperature=0.2,
    )


def test_ask_refuses_without_index(tmp_path: Path) -> None:
    pipeline = CampusRagPipeline(_config(tmp_path))
    result = pipeline.ask("请假怎么办？")

    assert result.refused is True
    assert "暂未在当前知识库中找到" in result.answer


def test_build_and_ask_with_text_knowledge(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "学生手册.txt").write_text(
        "第一章 学籍管理\n第十条 学生因病请假，应当履行请假手续。请假期满应及时销假。",
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)

    build_result = pipeline.build_index()
    answer = pipeline.ask("学生因病请假需要做什么？")

    assert build_result["chunks"] == 1
    assert answer.refused is False
    assert "请假手续" in answer.answer
    assert answer.hits[0].chunk.source == "学生手册.txt"


def test_question_subject_filters_conflicting_hits(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "培养方案.txt").write_text(
        "\n".join(
            [
                "第一章 课程学习计划",
                "博士生入学后应在2个月内按照学科（专业）培养方案制定课程学习计划。",
                "专业学位研究生入学后应在1个月内按学科（专业）培养方案制定课程学习计划。",
                "硕士生入学后应在1个月内依据学科（专业）培养方案制定课程学习计划。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("博士生入学后应在几个月内按照学科（专业）培养方案制定课程学习计划？")

    assert result.hits
    assert "博士生" in result.hits[0].chunk.text
    assert all("专业学位研究生" not in hit.chunk.text for hit in result.hits[:1])
    assert all("硕士生" not in hit.chunk.text for hit in result.hits[:1])


def test_precise_numeric_question_prefers_exact_sentence(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "开题要求.txt").write_text(
        "\n".join(
            [
                "六、开题",
                "专业学位硕士查阅文献一般不少于50篇，其中外文文献一般不少于10篇。",
                "学术学位硕士研究生查阅文献一般不少于80篇，其中外文文献一般不少于10篇。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("专业学位硕士查阅文献一般不少于多少篇？其中外文文献一般不少于多少篇？")

    assert "50篇" in result.answer
    assert "10篇" in result.answer
    assert "80篇" not in result.answer


def test_academic_activity_numeric_question_prefers_exact_values(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "学术活动要求.txt").write_text(
        "\n".join(
            [
                "八、学术活动",
                "研究生学习期间须参加各种学术活动，并填写学术活动记录表，记录学术活动内容和收获。",
                "各培养学院应明确研究生参加学术活动的总次数和本人主讲次数要求，其中学术学位硕士研究生要求至少公开做学术报告1次，或参加国内外学术会议1次。学术报告考核通过计2学分。",
                "开题报告通过，记1学分。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("学术学位硕士研究生至少公开做学术报告几次，或参加国内外学术会议几次？学术报告考核通过计几学分？")

    assert "学术报告1次" in result.answer
    assert "学术会议1次" in result.answer
    assert "2学分" in result.answer
    assert "开题报告通过，记1学分" not in result.answer


def test_wide_image_is_split_into_multiple_document_pages(tmp_path: Path) -> None:
    image_path = tmp_path / "双页拍照.jpg"
    image_path.write_bytes(b"fake-image")

    with patch("campus_rag.document_loader.extract_pages_from_image", return_value=["左页内容", "右页内容"]):
        pages = load_document(image_path)

    assert len(pages) == 2
    assert pages[0].source == "双页拍照.jpg"
    assert pages[0].page == 1
    assert pages[0].text == "左页内容"
    assert pages[1].page == 2
    assert pages[1].text == "右页内容"


def test_question_topic_filters_conflicting_hits(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "主题混合.txt").write_text(
        "\n".join(
            [
                "六、开题",
                "开题报告通过，记1学分。因特殊原因不能如期进行开题报告者，应提出书面申请。",
                "八、学术活动",
                "学术学位硕士研究生要求至少公开做学术报告1次，或参加国内外学术会议1次。学术报告考核通过计2学分。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("学术学位硕士研究生至少公开做学术报告几次，或参加国内外学术会议几次？学术报告考核通过计几学分？")

    assert result.hits
    assert result.hits[0].chunk.chapter_title == "八、学术活动"


def test_academic_activity_ocr_bang_is_treated_as_one_time(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "学术活动OCR.txt").write_text(
        "\n".join(
            [
                "八、学术活动",
                "研究生学习期间须参加各种学术活动，并填写学术活动记录表。",
                "各培养学院应明确研究生参加学术活动的总次数和本人主讲次数要求。",
                "次数要求，其中学术学位硕士研究生要求至少公开做学术报告！",
                "或参加国内外学术会议1次。学术报告考核通过计2学分。",
                "九、学位论文",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("学术学位硕士研究生至少公开做学术报告几次，或参加国内外学术会议几次？学术报告考核通过计几学分？")

    assert "学术报告1次" in result.answer
    assert "学术会议1次" in result.answer
    assert "2学分" in result.answer


def test_committee_size_question_prefers_people_count(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "中期考核要求.txt").write_text(
        "\n".join(
            [
                "七、中期考核",
                "次数要求，其中学术学位硕士研究生要求至少公开做学术报告1次，或参加国内外学术会议1次。学术报告考核通过计2学分。",
                "中期考核由各学院统一组织，硕士研究生考核小组不少于5人。研究生就中期考核内容做全面的自我总结，并向考核小组汇报。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("硕士研究生考核小组一般最少是几人？")

    assert "5人" in result.answer
    assert "学术报告1次" not in result.answer
    assert "2学分" not in result.answer


def test_precise_question_refuses_when_required_unit_is_missing(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "学术活动.txt").write_text(
        "八、学术活动\n学术学位硕士研究生要求至少公开做学术报告1次，或参加国内外学术会议1次。学术报告考核通过计2学分。",
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("硕士研究生考核小组一般最少是几人？")

    assert result.refused is False
    assert "暂未在当前知识库中找到" in result.answer
    assert "学术报告1次" not in result.answer
    assert "2学分" not in result.answer


def test_score_question_extracts_matching_grade_value(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "专业硕士合格标准.txt").write_text(
        "\n".join(
            [
                "4.申请专业硕士学位的研究生，其课程学习的合格标准是：",
                "（1）按课程学习计划修完全部课程，成绩合格。",
                "（2）必修课成绩平均不低于75分。",
                "（3）外语学位课成绩不低于60分。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("申请专业硕士学位的研究生，其课程学习的合格标准是必修课成绩平均不低于多少分。")

    assert "75分" in result.answer
    assert "60分" not in result.answer
    assert "暂未在当前知识库中找到" not in result.answer


def test_composite_precise_question_is_split_and_merged(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "博士学位论文外审.txt").write_text(
        "\n".join(
            [
                "四、学位论文外审前",
                "博士学位论文外审前，必须聘请至少3位专家对学位论文进行预审，预审时间不少于5个工作日。",
                "预审后方可进入后续外审流程。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("博士学位论文外审前，必须聘请至少几位专家对学位论文进行预审，预审时间不少于几个工作日？")

    assert "3位专家" in result.answer
    assert "5个工作日" in result.answer
    assert "暂未在当前知识库中找到" not in result.answer


def test_composite_precise_question_joins_values_from_noisy_separate_chunks(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.max_chars = 95
    cfg.overlap_chars = 0
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "博士学位论文外审OCR.txt").write_text(
        "\n".join(
            [
                "四、学位论文外审前",
                "1.博士学位论文外审前，必须聘请至少3位专家对学位论文",
                "第十二条学位论文质量是衡量研究生培养质量的重要指标。",
                "学院应组织预答辩工作，预答辩需在论文送审前1个月进行。",
                "进行预审，预审时间不少于5个工作日；预审结束后由所在学院组织后续工作。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("博士学位论文外审前，必须聘请至少几位专家对学位论文进行预审，预审时间不少于几个工作日？")

    assert "3位专家" in result.answer
    assert "5个工作日" in result.answer
    assert "1个月" not in result.answer
    assert "第十二条" not in result.answer
    assert "暂未在当前知识库中找到" not in result.answer


def test_composite_precise_question_can_join_different_units(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg.raw_dir.mkdir(parents=True)
    (cfg.raw_dir / "复合问题.txt").write_text(
        "\n".join(
            [
                "八、学术活动",
                "学术学位硕士研究生要求至少公开做学术报告1次，或参加国内外学术会议1次。学术报告考核通过计2学分。",
                "七、中期考核",
                "硕士研究生考核小组不少于5人。",
            ]
        ),
        encoding="utf-8",
    )
    pipeline = CampusRagPipeline(cfg)
    pipeline.build_index()

    result = pipeline.ask("学术学位硕士研究生至少公开做学术报告几次，或参加国内外学术会议几次？硕士研究生考核小组一般最少是几人？")

    assert "学术报告1次" in result.answer
    assert "学术会议1次" in result.answer
    assert "5人" in result.answer
