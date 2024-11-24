from fastapi import FastAPI
from pydantic import BaseModel, field_validator, computed_field, Field
from typing import List, Optional
from io import StringIO
import pandas as pd
import re
import pickle

app = FastAPI()
torque_val = re.compile(r'^(?P<value>[\d\.\,]+)')
max_torque_val = re.compile(r'.+\D(?P<value>\d{1,2}[\.,]?\d{3}).*$')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

cols = [
    'year',
    'km_driven',
    'mileage',
    'engine',
    'max_power',
    'torque',
    'seats',
    'max_torque_rpm'
]


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque_str: str = Field(validation_alias='torque')
    seats: float

    @field_validator('mileage', 'max_power', 'engine', mode='before')
    def to_float(value):
        if len(value.split()) == 2:
            return float(value.split()[0])

    @computed_field(repr=True, return_type=float)
    @property
    def torque(self) -> float:
        val = torque_val.search(self.torque_str).groupdict()['value'].replace(',', '.')

        if 'kgm' in self.torque_str.lower() and not 'nm' in self.torque_str.lower():
            return float(val) * 10
        return float(val)

    @computed_field(repr=True, return_type=float)
    @property
    def max_torque_rpm(self):
        val = max_torque_val.search(self.torque_str)
        if val:
            return float(val.groupdict()['value'].replace(',', ''))

    def selling_price(self) -> float:
        val = [self.dict()[col] for col in cols]
        pred = model.predict(scaler.transform([val]))
        return pred[0]

class Table(BaseModel):
    table: str


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return item.selling_price()


@app.post("/predict_items")
def predict_items(table: Table) -> str:
    df = pd.read_csv(StringIO(table.table))
    df['selling_price'] = [Item(**row).selling_price() for row in df.to_dict('records')]
    return df.to_csv()
