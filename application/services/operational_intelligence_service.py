# application/services/operation_intelligence_service.py

from domain.contracts.i_foundry_client import IFoundryClient
from typing import Dict, Any, Optional, List

class OperationalIntelligenceService:
    """
    Service for delivering operational insights, forecasts, and real-time alerts via Azure Foundry.
    """

    def __init__(self, foundry_client: IFoundryClient, default_nodel: Optional[str] = None):
        self.foundry_client = foundry_client
        self.default_model = default_nodel

    async def detect_anomalies(
        self,
        metrics: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        sensitivity: Optional[float] = 0.5,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Detects anomalies in time-series or operational metrics.
        Args:
            metrics: List of metrics or events (each can be a dict of values per timestamp).
            model: Optional model/deployment name for Foundry endpoint.
            sensitivity: Controls strictness of anomaly detection.
            additional_params: Any extra request parameters.
        Returns:
            Dict with anomaly report (could include indices, scores, or classified alerts).
        """
        payload = {
            "metrics": metrics,
            "sensitivity": sensitivity
        }
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="ops/anomaly-detection",  # Adapt this route to your actual Foundry deployment
            body=payload,
            model=model,
        )
        return response

    async def forecast_metrics(
        self,
        historical_data: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        forecast_horizon: int = 7,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Forecast future performance or KPIs based on historical metrics.
        Args:
            historical_data: List of historical metric dictionaries.
            forecast_horizon: Number of time steps (e.g., days/weeks) to forecast.
            model: Optional model/deployment name.
            additional_params: Optional extra params for forecasting.
        Returns:
            Dict of forecasted values and confidence intervals.
        """
        payload = {
            "historical_data": historical_data,
            "forecast_horizon": forecast_horizon
        }
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="ops/forecast-metrics",  # Route should match your Foundry API spec
            body=payload,
            model=model
        )
        return response

    async def operational_alerts(
        self,
        event_stream: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        alert_types: Optional[List[str]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates real-time or batch alerts based on operational events.
        Args:
            event_stream: List of operational events.
            alert_types: List of alert categories to generate/filter.
            model: Optional model/deployment name.
        Returns:
            List of alert dictionaries.
        """
        payload = {
            "event_stream": event_stream
        }
        if alert_types:
            payload["alert_types"] = alert_types
        if additional_params:
            payload.update(additional_params)

        response = await self.foundry_client.invoke(
            route="ops/generate-alerts",
            body=payload,
            model=model
        )
        return response.get("alerts", [])
