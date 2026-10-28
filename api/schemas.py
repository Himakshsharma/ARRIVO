"""Pydantic request and response models for the API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.geo import HUBS


class ShipmentIn(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "shipment_id": "ARV-DEMO-1", "origin_city": "Mumbai", "destination_city": "Kolkata",
        "dispatch_time": "2025-07-20T18:30:00", "vehicle_type": "32 FT Multi-Axle",
        "material_type": "Cement & Steel", "supplier_id": "SUP007"}})

    shipment_id: Optional[str] = Field(None, max_length=64)
    origin_city: str = Field(..., min_length=2, max_length=60)
    destination_city: str = Field(..., min_length=2, max_length=60)
    dispatch_time: datetime
    vehicle_type: Optional[str] = Field(None, max_length=60)
    material_type: Optional[str] = Field(None, max_length=60)
    supplier_id: Optional[str] = Field(None, max_length=30)
    # Optional overrides (otherwise looked up / estimated automatically)
    origin_lat: Optional[float] = Field(None, ge=6, le=37.5)       # India's bounding box
    origin_lon: Optional[float] = Field(None, ge=68, le=97.5)
    destination_lat: Optional[float] = Field(None, ge=6, le=37.5)
    destination_lon: Optional[float] = Field(None, ge=68, le=97.5)
    road_distance_km: Optional[float] = Field(None, gt=0, le=5000)
    planned_hours: Optional[float] = Field(None, gt=0, le=400)
    origin_rain_mm: Optional[float] = Field(None, ge=0, le=500)
    dest_rain_mm: Optional[float] = Field(None, ge=0, le=500)
    origin_temp_c: Optional[float] = Field(None, ge=-30, le=55)
    dest_temp_c: Optional[float] = Field(None, ge=-30, le=55)
    max_wind_kmph: Optional[float] = Field(None, ge=0, le=250)

    @field_validator("origin_city", "destination_city")
    @classmethod
    def tidy_city(cls, v: str) -> str:
        return v.strip().title()

    @model_validator(mode="after")
    def check_route(self):
        if self.origin_city == self.destination_city:
            raise ValueError("origin_city and destination_city must be different")
        for side, lat, lon in (("origin", self.origin_lat, self.origin_lon),
                               ("destination", self.destination_lat, self.destination_lon)):
            city = getattr(self, f"{side}_city")
            if city not in HUBS and (lat is None or lon is None):
                raise ValueError(f"'{city}' is not a supported hub; provide {side}_lat and {side}_lon "
                                 f"or use one of: {', '.join(sorted(HUBS))}")
        return self

    def to_record(self) -> dict:
        d = self.model_dump()
        d["dispatch_time"] = self.dispatch_time.replace(tzinfo=None).isoformat()
        return d


class Reason(BaseModel):
    factor: str
    text: str
    eta_impact_hours: float
    risk_direction: Literal["raises risk", "lowers risk"]
    risk_impact: float


class PredictionOut(BaseModel):
    shipment_id: str
    origin_city: str
    destination_city: str
    dispatch_time: str
    eta_hours: float
    expected_arrival: str
    planned_hours: float
    planned_arrival: str
    expected_delay_hours: float
    delay_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    flagged_late: bool
    top_reasons: list[Reason]
    route: dict
    context: dict


class BatchRowError(BaseModel):
    row: int
    shipment_id: Optional[str] = None
    error: str


class BatchOut(BaseModel):
    count: int
    high_risk: int
    medium_risk: int
    low_risk: int
    results: list[PredictionOut]
    errors: list[BatchRowError]


class AlertIn(BaseModel):
    shipment: ShipmentIn
    customer_name: str = Field("Customer", max_length=80)
    tone: str = Field("friendly and professional", max_length=60)
    use_llm: Optional[bool] = None


class AlertOut(BaseModel):
    message: str
    mode: Literal["llm", "template"]
    prediction: PredictionOut
