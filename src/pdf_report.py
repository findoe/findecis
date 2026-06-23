from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.agent import AgentReport, IndicatorAdvice
from src.config import PREDICTION_FULL_LABELS, PREDICTION_LABELS
from src.report_data import HistoryPoint


PAGE_TABLE_WIDTH = 500

REPORT = {
    "dark": "#0F172A",
    "dark_2": "#111827",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "violet": "#7C3AED",
    "pink": "#EC4899",
    "green": "#14B8A6",
    "amber": "#F59E0B",
    "red": "#E11D48",
    "text": "#111827",
    "muted": "#64748B",
    "line": "#E5EAF2",
    "soft": "#F1F5F9",
    "soft_blue": "#EEF2FF",
}


def create_pdf_report(
    file_path: str | Path,
    agent_report: AgentReport,
    context: dict[str, str],
    current_regression_values: Sequence[float],
    history_points: Sequence[HistoryPoint] | None = None,
) -> Path:
    output_path = Path(file_path)
    if output_path.suffix.lower() != ".pdf":
        output_path = output_path.with_suffix(".pdf")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer

    font_regular, font_bold = _register_report_fonts()
    _configure_matplotlib_font()

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "RuTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor(REPORT["dark"]),
        spaceAfter=10,
    )
    heading = ParagraphStyle(
        "RuHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=14,
        leading=17.5,
        textColor=colors.HexColor(REPORT["dark"]),
        spaceBefore=10,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="Отчет финансового агента",
        author="Financial decision support agent",
    )

    story: list = []
    story.append(_build_top_panel(context, agent_report, font_regular, font_bold))
    story.append(Spacer(1, 0.20 * cm))
    story.append(_build_action_card(agent_report, font_regular, font_bold))

    story.append(Paragraph("Прогнозные показатели", heading))
    story.append(_build_prediction_table(current_regression_values, font_regular, font_bold))

    advice_blocks = _build_advice_blocks(agent_report, font_regular, font_bold)
    if advice_blocks:
        story.append(KeepTogether([Paragraph("Интерпретация агента", heading), advice_blocks[0]]))
        story.extend(advice_blocks[1:])

    with TemporaryDirectory() as temp_dir:
        chart_paths = _build_charts(Path(temp_dir), current_regression_values, history_points or [])

        if chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("Графический анализ", title))
            for chart_title, chart_path in chart_paths:
                story.append(
                    KeepTogether(
                        [
                            Paragraph(chart_title, heading),
                            Image(str(chart_path), width=17.0 * cm, height=8.7 * cm),
                            Spacer(1, 0.35 * cm),
                        ]
                    )
                )
        else:
            story.append(Paragraph("Графический анализ", heading))
            story.append(
                _build_note_card(
                    "Для динамики нужен выбранный ИНН и история предприятия за несколько лет.",
                    font_regular,
                    font_bold,
                )
            )

        doc.build(story, onFirstPage=_add_page_number(font_regular), onLaterPages=_add_page_number(font_regular))

    return output_path


def _register_report_fonts() -> tuple[str, str]:
    from matplotlib import font_manager
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_path = font_manager.findfont("Segoe UI", fallback_to_default=True)
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="Segoe UI", weight="bold"),
        fallback_to_default=True,
    )

    regular_name = Path(regular_path).name.lower()
    bold_name = Path(bold_path).name.lower()
    if "segoeui" not in regular_name:
        regular_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    if "segoeui" not in bold_name:
        bold_path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
            fallback_to_default=True,
        )

    pdfmetrics.registerFont(TTFont("ReportRegular", regular_path))
    pdfmetrics.registerFont(TTFont("ReportBold", bold_path))
    return "ReportRegular", "ReportBold"


def _configure_matplotlib_font() -> None:
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _add_page_number(font_name: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColorRGB(0.45, 0.50, 0.58)
        canvas.setFont(font_name, 8)
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.75 * 28.3465, f"Страница {doc.page}")
        canvas.restoreState()

    return draw


def _build_top_panel(context: dict[str, str], agent_report: AgentReport, font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    hero_title = ParagraphStyle("TopHeroTitle", fontName=font_bold, fontSize=22, leading=27, textColor=colors.white)
    hero_subtitle = ParagraphStyle("TopHeroSubtitle", fontName=font_regular, fontSize=10.8, leading=14.2, textColor=colors.HexColor("#CBD5E1"))
    meta_style = ParagraphStyle("TopHeroMeta", fontName=font_bold, fontSize=10.2, leading=12.8, textColor=colors.HexColor("#E0F2FE"))

    metrics = _build_metric_cards(agent_report, font_regular, font_bold)
    rows = [
        [Paragraph("Отчет AI-агента", hero_title)],
        [Paragraph("Прогноз финансового состояния предприятия и вероятности банкротства", hero_subtitle)],
        [Paragraph(_hero_meta_html(context), meta_style)],
        [metrics],
    ]
    table = Table(rows, colWidths=[PAGE_TABLE_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(REPORT["dark"])),
                ("BOX", (0, 0), (-1, -1), 0.1, colors.HexColor(REPORT["dark"])),
                ("LEFTPADDING", (0, 0), (-1, 2), 16),
                ("RIGHTPADDING", (0, 0), (-1, 2), 16),
                ("LEFTPADDING", (0, 3), (-1, 3), 12),
                ("RIGHTPADDING", (0, 3), (-1, 3), 12),
                ("TOPPADDING", (0, 0), (-1, 0), 14),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
                ("TOPPADDING", (0, 2), (-1, 2), 0),
                ("BOTTOMPADDING", (0, 2), (-1, 2), 12),
                ("TOPPADDING", (0, 3), (-1, 3), 0),
                ("BOTTOMPADDING", (0, 3), (-1, 3), 12),
            ]
        )
    )
    return table


def _build_metric_cards(agent_report: AgentReport, font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    risk_color = _risk_hex(agent_report.risk_level)
    label = ParagraphStyle("MetricLabel", fontName=font_bold, fontSize=9.2, leading=11, textColor=colors.HexColor("#94A3B8"))
    value = ParagraphStyle("MetricValue", fontName=font_bold, fontSize=17, leading=20, textColor=colors.white)
    value_risk = ParagraphStyle("MetricValueRisk", parent=value, textColor=colors.HexColor(risk_color))
    value_probability = ParagraphStyle("MetricValueProbability", parent=value, textColor=colors.HexColor(risk_color))
    value_weak = ParagraphStyle("MetricValueWeak", parent=value, textColor=colors.HexColor(risk_color))

    rows = [
        [
            Paragraph("УРОВЕНЬ РИСКА", label),
            Paragraph("ВЕРОЯТНОСТЬ", label),
            Paragraph("СЛАБЫЕ БЛОКИ", label),
        ],
        [
            Paragraph(agent_report.risk_level.upper(), value_risk),
            Paragraph(f"{agent_report.probability * 100:.2f}%", value_probability),
            Paragraph(str(len(agent_report.weak_blocks)), value_weak),
        ],
    ]
    table = Table(rows, colWidths=[156, 156, 156])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(REPORT["dark"])),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor(REPORT["dark"])),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#1E293B")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ]
        )
    )
    return table


def _build_action_card(agent_report: AgentReport, font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    title = ParagraphStyle("ActionTitle", fontName=font_bold, fontSize=12, leading=14.5, textColor=colors.HexColor(REPORT["dark"]))
    text = ParagraphStyle("ActionText", fontName=font_regular, fontSize=10.5, leading=13.4, textColor=colors.HexColor(REPORT["text"]))
    rows = [
        [Paragraph("Итоговая оценка", title)],
        [Paragraph(_html_text(_format_summary_text(agent_report.risk_summary)), text)],
        [Paragraph(_html_text(agent_report.risk_action), text)],
    ]
    table = Table(rows, colWidths=[PAGE_TABLE_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(REPORT["soft_blue"])),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D2FE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _build_prediction_table(values: Sequence[float], font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    head = ParagraphStyle("PredictionHead", fontName=font_bold, fontSize=10.5, leading=13, textColor=colors.white)
    cell = ParagraphStyle("PredictionCell", fontName=font_bold, fontSize=10.5, leading=13, textColor=colors.HexColor(REPORT["text"]))
    rows = [[Paragraph("Показатель", head), Paragraph("Значение", head)]]

    for index, value_ in enumerate(values):
        score_color = _score_hex(float(value_))
        value_style = ParagraphStyle(
            f"PredictionValue{index}",
            fontName=font_bold,
            fontSize=10.5,
            leading=13,
            alignment=2,
            textColor=colors.HexColor(score_color),
        )
        rows.append(
            [
                Paragraph(_format_prediction_html(PREDICTION_FULL_LABELS[index]), cell),
                Paragraph(f"<b>{float(value_):.3f}</b>", value_style),
            ]
        )

    table = Table(rows, colWidths=[395, 105], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(REPORT["dark"])),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(REPORT["soft"])]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor(REPORT["line"])),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ]
        )
    )
    return table


def _build_advice_blocks(agent_report: AgentReport, font_regular: str, font_bold: str) -> list:
    return [
        _build_indicator_text_block(
            "Слабые блоки",
            agent_report.weak_blocks,
            font_regular,
            font_bold,
            title_color=REPORT["red"],
            accent_color=REPORT["red"],
            show_recommendation=True,
            empty_text="Не выявлены по установленному порогу.",
        ),
        _build_indicator_text_block(
            "Сильные блоки",
            agent_report.strong_blocks,
            font_regular,
            font_bold,
            title_color=REPORT["green"],
            accent_color=REPORT["green"],
            show_recommendation=False,
            empty_text="Не выявлены по установленному порогу.",
        ),
        _build_text_block("Рекомендации", agent_report.recommendations, font_regular, font_bold),
    ]


def _build_indicator_text_block(
    title: str,
    blocks: Sequence[IndicatorAdvice],
    font_regular: str,
    font_bold: str,
    title_color: str,
    accent_color: str,
    show_recommendation: bool,
    empty_text: str,
):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    title_style = ParagraphStyle(
        f"BlockTitle{title}",
        fontName=font_bold,
        fontSize=11.2,
        leading=13.8,
        textColor=colors.HexColor(title_color),
    )
    line_style = ParagraphStyle(
        f"BlockLine{title}",
        fontName=font_regular,
        fontSize=10.5,
        leading=13.2,
        textColor=colors.HexColor(REPORT["text"]),
    )
    rows = [[Paragraph(title, title_style)]]

    if not blocks:
        rows.append([Paragraph(_html_text(empty_text), line_style)])
    else:
        for block in blocks:
            prefix = f"{block.name}: {block.value:.3f}"
            suffix = f". Рекомендация: {block.recommendation}." if show_recommendation else "."
            line = (
                f"<font name='{font_bold}' color='{accent_color}'>{escape(prefix)}</font>{_html_text(suffix)}"
            )
            rows.append([Paragraph(line, line_style)])

    table = Table(rows, colWidths=[PAGE_TABLE_WIDTH], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(REPORT["dark"])),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor(REPORT["line"])),
                ("LINEBELOW", (0, 0), (-1, 0), 0.35, colors.HexColor(REPORT["dark"])),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6.5),
            ]
        )
    )
    return table


def _build_text_block(title: str, lines: Sequence[str], font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    title_style = ParagraphStyle("BlockTitle" + title, fontName=font_bold, fontSize=11.2, leading=13.8, textColor=colors.white)
    line_style = ParagraphStyle("BlockLine" + title, fontName=font_regular, fontSize=10.5, leading=13.2, textColor=colors.HexColor(REPORT["text"]))
    rows = [[Paragraph(title, title_style)]]
    rows.extend([[Paragraph(_html_text(line), line_style)] for line in lines])
    table = Table(rows, colWidths=[PAGE_TABLE_WIDTH], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(REPORT["dark"])),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor(REPORT["line"])),
                ("LINEBELOW", (0, 0), (-1, 0), 0.35, colors.HexColor(REPORT["dark"])),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6.5),
            ]
        )
    )
    return table


def _build_note_card(text: str, font_regular: str, font_bold: str):
    return _build_text_block("Примечание", [text], font_regular, font_bold)


def _build_charts(output_dir: Path, current_regression_values: Sequence[float], history_points: Sequence[HistoryPoint]) -> list[tuple[str, Path]]:
    chart_paths: list[tuple[str, Path]] = []
    prepared_history = _prepare_history(history_points)

    if len(prepared_history) >= 2:
        risk_path = output_dir / "risk_history.png"
        _save_risk_history_chart(prepared_history, risk_path)
        chart_paths.append(("Динамика вероятности банкротства", risk_path))

        stability_path = output_dir / "stability_history.png"
        _save_stability_history_chart(prepared_history, stability_path)
        chart_paths.append(("Динамика финансовой устойчивости", stability_path))

    blocks_path = output_dir / "current_blocks.png"
    _save_current_blocks_chart(current_regression_values, blocks_path)
    chart_paths.append(("Текущая оценка прогнозных блоков", blocks_path))

    return chart_paths


def _prepare_history(history_points: Sequence[HistoryPoint]) -> list[HistoryPoint]:
    def sort_key(point: HistoryPoint) -> tuple[int, str]:
        try:
            return int(float(str(point.year).replace(",", "."))), str(point.year)
        except ValueError:
            return 999999, str(point.year)

    return sorted(history_points, key=sort_key)


def _save_risk_history_chart(points: Sequence[HistoryPoint], output_path: Path) -> None:
    years = [str(point.year) for point in points]
    probabilities = [point.probability * 100 for point in points]

    fig, ax = plt.subplots(figsize=(8.3, 4.2), dpi=160)
    _style_chart(ax)
    ax.plot(years, probabilities, marker="o", linewidth=2.4, color=REPORT["blue"])
    ax.fill_between(years, probabilities, color=REPORT["blue"], alpha=0.12)
    ax.set_ylim(0, 100)
    ax.set_title("Вероятность банкротства по годам", fontsize=12, fontweight="bold")
    ax.set_xlabel("Год")
    ax.set_ylabel("Вероятность, %")

    for year, probability in zip(years, probabilities):
        ax.annotate(f"{probability:.1f}%", (year, probability), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_stability_history_chart(points: Sequence[HistoryPoint], output_path: Path) -> None:
    years = [str(point.year) for point in points]
    series = [
        ("Kredit", [point.regression_values[1] for point in points], REPORT["blue"]),
        ("Ability", [point.regression_values[6] for point in points], REPORT["green"]),
        ("Finn", [point.regression_values[8] for point in points], REPORT["amber"]),
        ("Z35", [point.regression_values[10] for point in points], REPORT["pink"]),
    ]

    fig, ax = plt.subplots(figsize=(8.3, 4.2), dpi=160)
    _style_chart(ax)
    for label, values, color in series:
        ax.plot(years, values, marker="o", linewidth=2.2, label=label, color=color)

    ax.set_ylim(0, 1)
    ax.set_title("Kredit, Ability, Finn и Z35 по годам", fontsize=12, fontweight="bold")
    ax.set_xlabel("Год")
    ax.set_ylabel("Оценка, 0-1")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_current_blocks_chart(values: Sequence[float], output_path: Path) -> None:
    numeric_values = [float(value) for value in values]
    sorted_indexes = sorted(range(len(numeric_values)), key=lambda index: numeric_values[index])
    labels = [PREDICTION_LABELS[index] for index in sorted_indexes]
    sorted_values = [numeric_values[index] for index in sorted_indexes]

    fig_height = max(4.4, 0.35 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.3, fig_height), dpi=160)
    _style_chart(ax)
    y_positions = np.arange(len(labels))
    colors = [REPORT["red"] if value < 0.4 else REPORT["amber"] if value < 0.7 else REPORT["green"] for value in sorted_values]
    ax.barh(y_positions, sorted_values, color=colors, alpha=0.9)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 1)
    ax.set_title("Прогнозные блоки: от слабых к сильным", fontsize=12, fontweight="bold")
    ax.set_xlabel("Оценка, 0-1")

    for y_position, value in zip(y_positions, sorted_values):
        ax.text(min(value + 0.025, 0.97), y_position, f"{value:.2f}", va="center", fontsize=8, color=REPORT["text"])

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_chart(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, alpha=0.35, color="#CBD5E1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(colors=REPORT["muted"], labelsize=8.5)
    ax.title.set_color(REPORT["dark"])
    ax.xaxis.label.set_color(REPORT["muted"])
    ax.yaxis.label.set_color(REPORT["muted"])


def _risk_hex(risk_level: str) -> str:
    if risk_level == "высокий":
        return REPORT["red"]
    if risk_level == "средний":
        return REPORT["amber"]
    return REPORT["green"]


def _score_hex(value: float) -> str:
    if value < 0.4:
        return REPORT["red"]
    if value < 0.7:
        return REPORT["amber"]
    return REPORT["green"]


def _hero_meta_html(context: dict[str, str]) -> str:
    meta_lines = [f"{escape(str(key))}: {escape(str(value))}" for key, value in context.items() if value]
    if not meta_lines:
        meta_lines.append("Источник: ручной ввод")
    meta_lines.append(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    return "<br/>".join(meta_lines)


def _format_prediction_html(text: str) -> str:
    formatted = escape(text)
    if "(" in text and ")" in text:
        left, rest = text.split("(", 1)
        middle, right = rest.split(")", 1)
        formatted = f"{escape(left).rstrip()} <b>({escape(middle)})</b>{escape(right)}"
    formatted = formatted.replace("Z25", "<b>Z25</b>").replace("Z35", "<b>Z35</b>")
    return formatted


def _format_summary_text(text: str) -> str:
    marker = "Наиболее слабые направления:"
    if marker in text:
        return text.replace(f" {marker}", f"\n{marker}")
    return text


def _html_text(text: str) -> str:
    return escape(str(text)).replace("\n", "<br/>")
