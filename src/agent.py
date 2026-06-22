from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from src.config import PREDICTION_FULL_LABELS

# =========================
#         АГЕНТ
# =========================

#Пороговые значения для интерпретации результата модели
LOW_RISK_LIMIT = 0.30
MEDIUM_RISK_LIMIT = 0.60
WEAK_SCORE_LIMIT = 0.40
STRONG_SCORE_LIMIT = 0.70



#Описание одного слабого или сильного блока
@dataclass(frozen=True)
class IndicatorAdvice:
    name: str
    value: float
    recommendation: str


#Итоговый отчет агента
@dataclass(frozen=True)
class AgentReport:
    risk_level: str
    risk_summary: str
    risk_action: str
    probability: float
    weak_blocks: list[IndicatorAdvice]
    strong_blocks: list[IndicatorAdvice]
    recommendations: list[str]
    report_text: str



#Рекомендации по каждому из 11 выходных регрессионных показателей
RECOMMENDATIONS_BY_INDEX = [
    "проверить влияние региональных и отраслевых рисков, а также сопоставить предприятие со средними значениями по отрасли",
    "оценить кредитную нагрузку, историю просрочек, структуру залогового обеспечения и условия действующих займов",
    "проанализировать состояние оборудования, загрузку мощностей и потребность в технологическом обновлении",
    "проверить зависимость от ключевых поставщиков и покупателей, динамику спроса и ценовую устойчивость",
    "оценить достаточность персонала, квалификацию сотрудников и кадровые риски",
    "проанализировать управленческий климат, устойчивость команды и влияние текучести на финансовые показатели",
    "проверить текущую ликвидность, обеспеченность собственными оборотными средствами и краткосрочные обязательства",
    "разобрать дебиторскую и кредиторскую задолженность, сроки оборота запасов и готовой продукции",
    "оценить долю заемного капитала, финансовую независимость и обеспеченность запасов собственными средствами",
    "сопоставить интегральный показатель Z25 с динамикой базовых финансовых коэффициентов",
    "сопоставить интегральный показатель Z35 с полным набором факторов и проверить блоки с минимальными оценками",
]



#Действия агента для каждого уровня риска
RISK_ACTIONS = {
    "низкий": "Критических признаков дефолта по расчету модели не выявлено. Достаточно планового мониторинга ключевых финансовых коэффициентов.",
    "средний": "Нужен углубленный анализ слабых блоков: предприятие находится в зоне неопределенности, где отдельные ухудшения могут быстро повысить риск.",
    "высокий": "Требуется приоритетная финансовая диагностика: ликвидность, долговая нагрузка, оборачиваемость и финансовая устойчивость должны быть проверены в первую очередь.",
}



#Формирование полного текстового и структурированного отчета
def build_agent_report(
    regression_values: Sequence[float],
    probability: float,
    context: dict[str, str] | None = None,
) -> AgentReport:
    values = [float(value) for value in regression_values]
    risk_level = classify_risk(probability)
    weak_blocks = _find_weak_blocks(values)
    strong_blocks = _find_strong_blocks(values)
    recommendations = _build_recommendations(risk_level, weak_blocks)
    risk_summary = _build_risk_summary(risk_level, probability, weak_blocks)
    report_text = _build_report_text(
        risk_level=risk_level,
        probability=probability,
        values=values,
        context=context or {},
        weak_blocks=weak_blocks,
        strong_blocks=strong_blocks,
        recommendations=recommendations,
    )

    return AgentReport(
        risk_level=risk_level,
        risk_summary=risk_summary,
        risk_action=RISK_ACTIONS[risk_level],
        probability=probability,
        weak_blocks=weak_blocks,
        strong_blocks=strong_blocks,
        recommendations=recommendations,
        report_text=report_text,
    )


#Классификация риска по вероятности банкротства
def classify_risk(probability: float) -> str:
    if probability < LOW_RISK_LIMIT:
        return "низкий"
    if probability < MEDIUM_RISK_LIMIT:
        return "средний"
    return "высокий"


#Поиск слабых направлений
def _find_weak_blocks(values: list[float]) -> list[IndicatorAdvice]:
    weak_indexes = [
        index for index, value in enumerate(values)
        if value < WEAK_SCORE_LIMIT
    ]
    weak_indexes.sort(key=lambda index: values[index])

    return [
        IndicatorAdvice(
            name=PREDICTION_FULL_LABELS[index],
            value=values[index],
            recommendation=RECOMMENDATIONS_BY_INDEX[index],
        )
        for index in weak_indexes[:5]
    ]


#Поиск сильных направлений
def _find_strong_blocks(values: list[float]) -> list[IndicatorAdvice]:
    strong_indexes = [
        index for index, value in enumerate(values)
        if value >= STRONG_SCORE_LIMIT
    ]
    strong_indexes.sort(key=lambda index: values[index], reverse=True)

    return [
        IndicatorAdvice(
            name=PREDICTION_FULL_LABELS[index],
            value=values[index],
            recommendation="использовать как опорный фактор при общей оценке финансового состояния",
        )
        for index in strong_indexes[:3]
    ]



#Сбор итогового списка рекомендаций
def _build_recommendations(risk_level: str, weak_blocks: list[IndicatorAdvice]) -> list[str]:
    recommendations: list[str] = []

    if risk_level == "высокий":
        recommendations.append(
            "Сначала проверить платежеспособность и финансовую устойчивость: именно эти блоки чаще всего требуют быстрых управленческих решений."
        )
    elif risk_level == "средний":
        recommendations.append(
            "Провести дополнительную проверку слабых блоков и сравнить показатели с предыдущими годами по тому же ИНН."
        )
    else:
        recommendations.append(
            "Сохранить плановый мониторинг и повторять расчет при появлении новых отчетных данных."
        )

    for block in weak_blocks[:4]:
        recommendations.append(f"По блоку «{block.name}» — {block.recommendation}.")

    if not weak_blocks:
        recommendations.append(
            "Явных слабых блоков по прогнозным показателям не обнаружено; стоит контролировать динамику вероятности банкротства."
        )

    return recommendations[:6]



#Краткое резюме результата анализа
def _build_risk_summary(risk_level: str, probability: float, weak_blocks: list[IndicatorAdvice]) -> str:
    weak_part = ""
    if weak_blocks:
        weak_names = ", ".join(block.name.split(" (")[0] for block in weak_blocks[:3])
        weak_part = f" Наиболее слабые направления: {weak_names}."

    return (
        f"Агент классифицирует риск как {risk_level}. "
        f"Расчетная вероятность банкротства составляет {probability * 100:.2f}%."
        f"{weak_part}"
    )



#Формирование текстового отчета для сохранения в файл
def _build_report_text(
    risk_level: str,
    probability: float,
    values: list[float],
    context: dict[str, str],
    weak_blocks: list[IndicatorAdvice],
    strong_blocks: list[IndicatorAdvice],
    recommendations: list[str],
) -> str:
    lines: list[str] = [
        "ОТЧЕТ ФИНАНСОВОГО АГЕНТА",
        f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        "",
    ]

    if context:
        lines.append("Контекст анализа:")
        for key, value in context.items():
            if value:
                lines.append(f"- {key}: {value}")
        lines.append("")

    lines.extend([
        "Итоговая оценка:",
        f"- Уровень риска: {risk_level}",
        f"- Вероятность банкротства: {probability * 100:.2f}%",
        f"- Действие агента: {RISK_ACTIONS[risk_level]}",
        "",
        "Прогнозные показатели:",
    ])

    for name, value in zip(PREDICTION_FULL_LABELS, values):
        lines.append(f"- {name}: {value:.3f}")

    lines.append("")
    lines.append("Слабые блоки:")
    if weak_blocks:
        for block in weak_blocks:
            lines.append(f"- {block.name}: {block.value:.3f}. Рекомендация: {block.recommendation}.")
    else:
        lines.append("- Не выявлены по установленному порогу.")

    lines.append("")
    lines.append("Сильные блоки:")
    if strong_blocks:
        for block in strong_blocks:
            lines.append(f"- {block.name}: {block.value:.3f}.")
    else:
        lines.append("- Не выявлены по установленному порогу.")

    lines.append("")
    lines.append("Рекомендации:")
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")


    return "\n".join(lines)