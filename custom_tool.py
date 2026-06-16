from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Type, Optional
import os
import csv


# Defining input schema for our custom Read/Write (append) tool
class CSVToolInput(BaseModel):
    path: str = Field(..., description="The path to the CSV file (e.g., 'data/patients_data.csv')")
    action: str = Field(..., description="The action to perform: 'read' to fetch data, or 'append' to save a new patient.")
    name: Optional[str] = Field(None, description="Patient's full name")
    age: Optional[int] = Field(None, description="Patient's age")
    gender: Optional[str] = Field(None, description="Patient's gender")
    contact: Optional[str] = Field(None, description="Patient's contact number")
    medical_history: Optional[str] = Field(None, description="Patient's medical history")


# Defining custom tool
class CSVReadWriteTool(BaseTool):
    name: str = "CSV Database Reader and Writer Tool"
    description: str = "Useful for reading patient records or appending new patient registration details into a CSV file."
    args_schema: Type[BaseModel] = CSVToolInput


    def _run(self, path: str, action: str, name: str = None, age: int = None, 
             gender: str = None, contact: str = None, medical_history: str = None) -> str:
        headers = ['patient_id', 'name', 'age', 'gender', 'medical_history', 'contact']
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        if action == "read":
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return "File is empty or does not exist yet."
            with open(path, mode="r", encoding="utf-8-sig") as file:
                return str(list(csv.DictReader(file)))

        elif action == "append":
            next_id = 1
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, mode='r', newline='', encoding='utf-8') as file:
                    reader = list(csv.DictReader(file))
                    if reader and reader[-1].get('patient_id'):
                        next_id = int(reader[-1]['patient_id']) + 1
            with open(path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    writer.writeheader()
                writer.writerow({'patient_id': str(next_id), 'name': name, 'age': age, 'gender': gender, 'medical_history': medical_history, 'contact': contact})
            return f"Success: Registered with ID: {next_id}"

        
