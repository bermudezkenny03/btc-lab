"""Laboratorio de predicción de Bitcoin, partido por responsabilidades (SOLID).

El comando sigue siendo ``python lab_btc.py``. Cada pieza se puede cambiar
o ampliar sin reescribir el resto:

- una fuente de velas nueva implementa ``CandleSource``;
- un bloque de variables nuevo implementa ``FeatureBuilder``;
- un modelo nuevo se registra en ``ModelCatalog``.
"""

__version__ = "0.2.0"
