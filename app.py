"""NexaLog Brasil — Dashboard Executivo de Logística."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="NexaLog Brasil | Inteligência Logística",
    page_icon="📦",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" / "processed"
    / "dashboard_validation.csv"
)

REPORT_PATH = (
    ROOT / "reports"
    / "final_holdout_evaluation.json"
)


# ============================================================
# FUNÇÕES
# ============================================================

def fmt_int(value):
    return f"{value:,.0f}".replace(",", ".")


def fmt_pct(value):
    return f"{value * 100:.2f}%".replace(".", ",")


def month_label(value):
    return {
        "2018-03": "Março/2018",
        "2018-04": "Abril/2018",
    }.get(value, value)


@st.cache_data
def load_data():

    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["prediction_timestamp"],
        dtype={"order_id": "string"},
    )

    required = [
        "order_id",
        "prediction_timestamp",
        "is_late",
        "promised_days",
        "remaining_days",
        "delay_risk_score",
        "reference_month",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Colunas ausentes: {missing}"
        )

    if data.empty:
        raise ValueError("Base histórica vazia.")

    if data[required].isna().any().any():
        raise ValueError(
            "Existem valores obrigatórios ausentes."
        )

    if data["order_id"].duplicated().any():
        raise ValueError(
            "Existem pedidos duplicados."
        )

    if not data["is_late"].isin([0, 1]).all():
        raise ValueError(
            "Existem rótulos históricos inválidos."
        )

    return data


@st.cache_data
def load_evaluation():

    with REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


try:
    df = load_data()
    evaluation = load_evaluation()

except (
    FileNotFoundError,
    ValueError,
    KeyError,
    json.JSONDecodeError,
) as error:

    st.error(
        f"Falha no carregamento: {error}"
    )

    st.stop()


# ============================================================
# CABEÇALHO
# ============================================================

st.title("📦 NexaLog Brasil")

st.subheader(
    "Inteligência e Desempenho Logístico"
)

st.caption(
    "Painel executivo de entregas | "
    "Análise histórica e indicadores de gestão"
)

st.info(
    "Este painel apresenta dados históricos "
    "de março e abril de 2018. "
    "Não representa uma operação em tempo real."
)


# ============================================================
# FILTROS
# ============================================================

with st.sidebar:

    st.header("Filtros")

    months = sorted(
        df["reference_month"].unique()
    )

    selected_months = st.multiselect(
        "Período analisado",
        options=months,
        default=months,
        format_func=month_label,
    )

    st.divider()

    st.caption(
        "Fonte: dados históricos de validação."
    )

    st.caption(
        "Indicadores de desempenho observados, "
        "sem previsões operacionais."
    )


if not selected_months:
    st.warning(
        "Selecione pelo menos um período."
    )
    st.stop()


filtered = df[
    df["reference_month"].isin(
        selected_months
    )
].copy()


# ============================================================
# 1. RESUMO EXECUTIVO
# ============================================================

total = len(filtered)

late = int(filtered["is_late"].sum())

on_time = total - late

late_rate = late / total if total else 0

on_time_rate = on_time / total if total else 0

st.divider()

st.header("1. Visão executiva")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Pedidos analisados",
    fmt_int(total),
)

c2.metric(
    "Entregas no prazo",
    fmt_int(on_time),
)

c3.metric(
    "Entregas atrasadas",
    fmt_int(late),
)

c4.metric(
    "Taxa de atraso",
    fmt_pct(late_rate),
)

st.caption(
    f"{fmt_pct(on_time_rate)} das entregas "
    "foram concluídas no prazo."
)


# ============================================================
# 2. EVOLUÇÃO MENSAL
# ============================================================

st.divider()

st.header("2. Evolução das entregas")

monthly = (
    filtered.groupby("reference_month")
    .agg(
        pedidos=("order_id", "size"),
        atrasados=("is_late", "sum"),
    )
    .reset_index()
    .sort_values("reference_month")
)

monthly["no_prazo"] = (
    monthly["pedidos"]
    - monthly["atrasados"]
)

monthly["taxa_atraso"] = (
    monthly["atrasados"]
    / monthly["pedidos"]
    * 100
)

monthly["mes"] = (
    monthly["reference_month"]
    .map(month_label)
)

left, right = st.columns(2)

with left:

    fig_rate = px.bar(
        monthly,
        x="mes",
        y="taxa_atraso",
        text="taxa_atraso",
        title="Taxa de atraso por mês",
        labels={
            "mes": "Mês",
            "taxa_atraso": "Atrasos (%)",
        },
    )

    fig_rate.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig_rate.update_layout(
        yaxis_rangemode="tozero",
    )

    st.plotly_chart(
        fig_rate,
        use_container_width=True,
    )

with right:

    volume = monthly[
        ["mes", "no_prazo", "atrasados"]
    ].melt(
        id_vars=["mes"],
        value_vars=["no_prazo", "atrasados"],
        var_name="situacao",
        value_name="quantidade",
    )

    volume["situacao"] = (
        volume["situacao"]
        .replace({
            "no_prazo": "No prazo",
            "atrasados": "Atrasados",
        })
    )

    fig_volume = px.bar(
        volume,
        x="mes",
        y="quantidade",
        color="situacao",
        barmode="stack",
        title="Volume de entregas",
        labels={
            "mes": "Mês",
            "quantidade": "Pedidos",
            "situacao": "Situação",
        },
    )

    st.plotly_chart(
        fig_volume,
        use_container_width=True,
    )


# ============================================================
# 3. ANÁLISE DOS PRAZOS
# ============================================================

st.divider()

st.header("3. Análise dos prazos logísticos")

st.write(
    "Comparação dos prazos registrados "
    "nos pedidos históricos."
)

p1, p2 = st.columns(2)

p1.metric(
    "Prazo prometido mediano",
    f"{filtered['promised_days'].median():.0f} dias",
)

p2.metric(
    "Prazo restante mediano na referência",
    f"{filtered['remaining_days'].median():.0f} dias",
)

st.caption(
    "O prazo restante refere-se à data de "
    "referência da previsão histórica. "
    "Não representa o tempo real de transporte."
)

deadline_summary = (
    filtered.groupby("reference_month")
    .agg(
        prazo_prometido=(
            "promised_days", "median"
        ),
        prazo_restante=(
            "remaining_days", "median"
        ),
    )
    .reset_index()
)

deadline_summary["mes"] = (
    deadline_summary["reference_month"]
    .map(month_label)
)

deadline_long = deadline_summary[
    ["mes", "prazo_prometido", "prazo_restante"]
].melt(
    id_vars="mes",
    var_name="indicador",
    value_name="dias",
)

deadline_long["indicador"] = (
    deadline_long["indicador"]
    .replace({
        "prazo_prometido": "Prazo prometido",
        "prazo_restante": "Prazo restante",
    })
)

fig_deadline = px.bar(
    deadline_long,
    x="mes",
    y="dias",
    color="indicador",
    barmode="group",
    title="Prazos medianos por mês",
    labels={
        "mes": "Mês",
        "dias": "Dias",
        "indicador": "Indicador",
    },
)

st.plotly_chart(
    fig_deadline,
    use_container_width=True,
)


# ============================================================
# 4. CONSULTA DE PEDIDOS
# ============================================================

st.divider()

st.header("4. Consulta de pedidos")

st.write(
    "Consulte os registros históricos e "
    "exporte os dados para análise."
)

status_filter = st.selectbox(
    "Situação da entrega",
    options=[
        "Todos",
        "No prazo",
        "Atrasados",
    ],
)

orders = filtered.copy()

if status_filter == "No prazo":
    orders = orders[
        orders["is_late"] == 0
    ]

elif status_filter == "Atrasados":
    orders = orders[
        orders["is_late"] == 1
    ]

orders["situacao"] = np.where(
    orders["is_late"] == 1,
    "Atrasou",
    "No prazo",
)

orders["data"] = (
    orders["prediction_timestamp"]
    .dt.strftime("%d/%m/%Y %H:%M")
)

orders_display = orders[
    [
        "order_id",
        "data",
        "promised_days",
        "remaining_days",
        "situacao",
    ]
].rename(
    columns={
        "order_id": "Pedido",
        "data": "Data de referência",
        "promised_days": "Prazo prometido (dias)",
        "remaining_days": "Prazo restante (dias)",
        "situacao": "Resultado histórico",
    }
)

st.dataframe(
    orders_display,
    hide_index=True,
    use_container_width=True,
    height=420,
)

st.download_button(
    "Exportar pedidos (CSV)",
    data=orders_display.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="nexalog_pedidos_historicos.csv",
    mime="text/csv",
)


# ============================================================
# 5. TRANSPARÊNCIA DO EXPERIMENTO
# ============================================================

st.divider()

st.header("5. Avaliação do experimento preditivo")

st.warning(
    "O modelo de previsão de atrasos "
    "não atingiu o desempenho mínimo "
    "definido para aprovação."
)

metrics = evaluation["metrics"]

e1, e2, e3 = st.columns(3)

e1.metric(
    "Pedidos do teste final",
    fmt_int(evaluation["total_orders"]),
)

e2.metric(
    "ROC-AUC",
    f"{metrics['roc_auc']:.4f}",
)

e3.metric(
    "Lift@10%",
    f"{metrics['lift_at_10']:.2f}x",
)

st.write(
    "O teste final foi realizado com dados "
    "temporais reservados. O ROC-AUC "
    "observado ficou abaixo do mínimo "
    "de 0,70 estabelecido previamente."
)

st.info(
    "Decisão técnica: o modelo permanece "
    "documentado como experimento, "
    "sem recomendação de utilização "
    "preditiva em produção."
)

st.caption(
    "NexaLog Brasil | Projeto de Ciência "
    "de Dados e Business Intelligence."
)
