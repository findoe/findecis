from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.agent import AgentReport
from src.config import FONT_FAMILY, PREDICTION_FULL_LABELS, PREDICTION_LABELS
from src.report_data import HistoryPoint


#Создание PDF-отчета финансового агента
#В отчет включаются текстовая интерпретация, таблица прогнозов и графики.
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
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    font_regular, font_bold = _register_report_fonts()
    _configure_matplotlib_font()

    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "RuNormal",
        parent=styles["Normal"],
        fontName=font_regular,
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
    )
    title = ParagraphStyle(
        "RuTitle",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=19,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading = ParagraphStyle(
        "RuHeading",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=13.5,
        leading=17,
        spaceBefore=10,
        spaceAfter=8,
    )
    small = ParagraphStyle(
        "RuSmall",
        parent=normal,
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#555555"),
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.4 * cm,
        title="Отчет финансового агента",
        author="Financial decision support agent",
    )

    story: list = []
    story.append(Paragraph("ОТЧЕТ ФИНАНСОВОГО АГЕНТА", title))
    story.append(Paragraph("Автоматически сформированный PDF-отчет по результатам нейросетевого анализа.", normal))

    story.append(Paragraph("Контекст анализа", heading))
    story.append(_build_context_table(context, font_regular, font_bold))

    story.append(Paragraph("Итоговая оценка", heading))
    risk_percent = agent_report.probability * 100
    story.append(
        _build_summary_table(
            agent_report.risk_level,
            risk_percent,
            agent_report.risk_action,
            font_regular,
            font_bold,
        )
    )
    story.append(Paragraph(_escape_text(agent_report.risk_summary), normal))

    story.append(Paragraph("Прогнозные показатели", heading))
    story.append(_build_prediction_table(current_regression_values, font_regular, font_bold))

    story.append(Paragraph("Интерпретация агента", heading))
    story.extend(_build_advice_blocks(agent_report, normal, heading))

    with TemporaryDirectory() as temp_dir:
        chart_paths = _build_charts(
            Path(temp_dir),
            current_regression_values,
            history_points or [],
        )

        if chart_paths:
            story.append(PageBreak())
            story.append(Paragraph("Графический анализ", title))
            for chart_title, chart_path in chart_paths:
                story.append(
                    KeepTogether(
                        [
                            Paragraph(chart_title, heading),
                            Image(str(chart_path), width=17.0 * cm, height=8.7 * cm),
                            Spacer(1, 0.4 * cm),
                        ]
                    )
                )
        else:
            story.append(Paragraph("Графический анализ", heading))
            story.append(
                Paragraph(
                    "Графики не построены: для динамики нужен выбранный ИНН и история предприятия за несколько лет.",
                    normal,
                )
            )

        doc.build(story, onFirstPage=_add_page_number(font_regular), onLaterPages=_add_page_number(font_regular))

    return output_path


#Регистрация шрифта с поддержкой кириллицы для ReportLab
#Файлы шрифтов не копируются в проект, используется шрифт из установленного matplotlib.
def _register_report_fonts() -> tuple[str, str]:
    from matplotlib import font_manager
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    regular_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
    bold_path = font_manager.findfont(
        font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
        fallback_to_default=True,
    )

    pdfmetrics.registerFont(TTFont("DejaVuSans", regular_path))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
    return "DejaVuSans", "DejaVuSans-Bold"


#Настройка matplotlib для корректной кириллицы на графиках
def _configure_matplotlib_font() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


#Нумерация страниц PDF
def _add_page_number(font_name: str):
    def draw(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.75 * 28.3465, f"Страница {doc.page}")
        canvas.restoreState()

    return draw


#Таблица с контекстом анализа
def _build_context_table(context: dict[str, str], font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    rows = [["Параметр", "Значение"]]
    if context:
        rows.extend([[str(key), str(value)] for key, value in context.items() if value])
    else:
        rows.append(["Источник", "ручной ввод"])

    table = Table(rows, colWidths=[130, 340])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_regular),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


#Таблица итогового риска
def _build_summary_table(risk_level: str, risk_percent: float, action: str, font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle

    style = ParagraphStyle("SummaryCell", fontName=font_regular, fontSize=9.7, leading=12)
    rows = [
        ["Уровень риска", risk_level],
        ["Вероятность банкротства", f"{risk_percent:.2f}%"],
        ["Действие агента", Paragraph(_escape_text(action), style)],
    ]
    table = Table(rows, colWidths=[185, 285])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), font_bold),
                ("FONTNAME", (1, 0), (1, 1), font_bold),
                ("FONTNAME", (1, 2), (1, 2), font_regular),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF2F2")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


#Таблица 11 прогнозных показателей
def _build_prediction_table(values: Sequence[float], font_regular: str, font_bold: str):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    rows = [["Показатель", "Значение"]]
    rows.extend(
        [PREDICTION_FULL_LABELS[index], f"{float(value):.3f}"]
        for index, value in enumerate(values)
    )

    table = Table(rows, colWidths=[390, 80], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_regular),
                ("FONTNAME", (1, 1), (1, -1), font_bold),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#264653")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D0D0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


#Текстовые блоки рекомендаций
def _build_advice_blocks(agent_report: AgentReport, normal, heading) -> list:
    from reportlab.platypus import Paragraph

    story: list = []
    story.append(Paragraph("Слабые блоки", heading))
    if agent_report.weak_blocks:
        for block in agent_report.weak_blocks:
            story.append(
                Paragraph(
                    f"- {_escape_text(block.name)}: {block.value:.3f}. Рекомендация: {_escape_text(block.recommendation)}.",
                    normal,
                )
            )
    else:
        story.append(Paragraph("- Не выявлены по установленному порогу.", normal))

    story.append(Paragraph("Сильные блоки", heading))
    if agent_report.strong_blocks:
        for block in agent_report.strong_blocks:
            story.append(Paragraph(f"- {_escape_text(block.name)}: {block.value:.3f}.", normal))
    else:
        story.append(Paragraph("- Не выявлены по установленному порогу.", normal))

    story.append(Paragraph("Рекомендации", heading))
    for recommendation in agent_report.recommendations:
        story.append(Paragraph(f"- {_escape_text(recommendation)}", normal))

    return story


#Построение набора графиков для отчета
def _build_charts(
    output_dir: Path,
    current_regression_values: Sequence[float],
    history_points: Sequence[HistoryPoint],
) -> list[tuple[str, Path]]:
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


#Подготовка исторических точек к сортировке по году
def _prepare_history(history_points: Sequence[HistoryPoint]) -> list[HistoryPoint]:
    def sort_key(point: HistoryPoint) -> tuple[int, str]:
        try:
            return int(float(str(point.year).replace(",", "."))), str(point.year)
        except ValueError:
            return 999999, str(point.year)

    return sorted(history_points, key=sort_key)


#График вероятности банкротства по годам
def _save_risk_history_chart(points: Sequence[HistoryPoint], output_path: Path) -> None:
    years = [str(point.year) for point in points]
    probabilities = [point.probability * 100 for point in points]

    fig, ax = plt.subplots(figsize=(8.3, 4.2), dpi=160)
    ax.plot(years, probabilities, marker="o", linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_title("Вероятность банкротства по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Вероятность, %")
    ax.grid(True, alpha=0.35)

    for year, probability in zip(years, probabilities):
        ax.annotate(f"{probability:.1f}%", (year, probability), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


#График Kredit, Ability, Finn и Z35 по годам
def _save_stability_history_chart(points: Sequence[HistoryPoint], output_path: Path) -> None:
    years = [str(point.year) for point in points]
    series = [
        ("Kredit", [point.regression_values[1] for point in points]),
        ("Ability", [point.regression_values[6] for point in points]),
        ("Finn", [point.regression_values[8] for point in points]),
        ("Z35", [point.regression_values[10] for point in points]),
    ]

    fig, ax = plt.subplots(figsize=(8.3, 4.2), dpi=160)
    for label, values in series:
        ax.plot(years, values, marker="o", linewidth=2, label=label)

    ax.set_ylim(0, 1)
    ax.set_title("Kredit, Ability, Finn и Z35 по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Оценка, 0-1")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


#График текущих прогнозных блоков
def _save_current_blocks_chart(values: Sequence[float], output_path: Path) -> None:
    numeric_values = [float(value) for value in values]
    sorted_indexes = sorted(range(len(numeric_values)), key=lambda index: numeric_values[index])
    labels = [PREDICTION_LABELS[index] for index in sorted_indexes]
    sorted_values = [numeric_values[index] for index in sorted_indexes]

    fig_height = max(4.4, 0.35 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.3, fig_height), dpi=160)
    y_positions = np.arange(len(labels))
    ax.barh(y_positions, sorted_values)
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 1)
    ax.set_title("Прогнозные блоки: от слабых к сильным")
    ax.set_xlabel("Оценка, 0-1")
    ax.grid(True, axis="x", alpha=0.35)

    for y_position, value in zip(y_positions, sorted_values):
        ax.text(min(value + 0.025, 0.97), y_position, f"{value:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


#Минимальное экранирование текста для Paragraph
def _escape_text(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
