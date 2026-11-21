"""Motor de inferência da NexaLog Brasil.

Recebe as 19 preditoras previamente construídas
e devolve scores de risco de atraso.

Carrega somente artefatos locais confiáveis.
"""

from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


class LogisticsInference:

    def __init__(self, project_root):

        root = Path(project_root).resolve()

        model_path = (
            root / "models" / "logistic_pipeline.joblib"
        )

        metadata_path = (
            root / "models" / "model_metadata.json"
        )

        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)

        with metadata_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.metadata = json.load(file)

        digest = hashlib.sha256()

        with model_path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b""
            ):
                digest.update(chunk)

        actual_hash = digest.hexdigest()

        expected_hash = self.metadata["model_sha256"]

        if actual_hash != expected_hash:
            raise ValueError(
                "Integridade do modelo comprometida."
            )

        # Carregar somente arquivos confiáveis.
        self.model = joblib.load(model_path)

        self.features = list(
            self.metadata["features"]
        )

        self.numeric_features = list(
            self.metadata["numeric_features"]
        )

        self.categorical_features = list(
            self.metadata["categorical_features"]
        )

        if len(self.features) != 19:
            raise ValueError(
                "Quantidade inesperada de preditoras."
            )

        if len(set(self.features)) != 19:
            raise ValueError(
                "Preditoras duplicadas nos metadados."
            )

        if (
            set(self.numeric_features)
            & set(self.categorical_features)
        ):
            raise ValueError(
                "Preditoras com tipos conflitantes."
            )

        if (
            set(self.numeric_features)
            | set(self.categorical_features)
        ) != set(self.features):
            raise ValueError(
                "Esquema de preditoras inconsistente."
            )

    def validate(self, data):

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "A entrada deve ser um DataFrame."
            )

        if data.empty:
            raise ValueError(
                "A entrada não pode estar vazia."
            )

        missing = [
            column
            for column in self.features
            if column not in data.columns
        ]

        unexpected = [
            column
            for column in data.columns
            if column not in self.features
        ]

        if missing:
            raise ValueError(
                f"Preditoras ausentes: {missing}"
            )

        if unexpected:
            raise ValueError(
                f"Preditoras inesperadas: {unexpected}"
            )

        validated = data.loc[
            :, self.features
        ].copy()

        for column in self.numeric_features:

            original = validated[column]

            converted = pd.to_numeric(
                original,
                errors="coerce"
            )

            invalid = (
                original.notna()
                & converted.isna()
            )

            if invalid.any():
                raise ValueError(
                    f"Valor não numérico: {column}"
                )

            finite_or_missing = (
                converted.isna()
                | np.isfinite(converted)
            )

            if not finite_or_missing.all():
                raise ValueError(
                    f"Valor infinito: {column}"
                )

            validated[column] = converted

        for column in self.categorical_features:

            validated[column] = (
                validated[column]
                .astype("string")
                .astype(object)
            )

            validated[column] = (
                validated[column]
                .where(
                    pd.notna(validated[column]),
                    np.nan
                )
            )

        return validated

    def predict(self, data):

        validated = self.validate(data)

        probabilities = self.model.predict_proba(
            validated
        )[:, 1]

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Probabilidades não finitas."
            )

        if not (
            (probabilities >= 0)
            & (probabilities <= 1)
        ).all():
            raise ValueError(
                "Probabilidades fora do intervalo [0, 1]."
            )

        return pd.DataFrame(
            {
                "delay_risk_score": probabilities
            },
            index=validated.index
        )
