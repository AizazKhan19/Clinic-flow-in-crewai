from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Type, Optional, Literal
import os
import csv
import json
import re


class CSVToolInput(BaseModel):
    path: str = Field(..., description="The path to the CSV file (e.g., 'data/patients_data.csv')")
    action: Literal['read', 'lookup', 'append'] = Field(
        ..., 
        description="The action to perform: 'read' to fetch a patient by ID or return all rows, 'lookup' to detect an ID in text then fetch it, or 'append' to save a new patient."
    )
    patient_id: Optional[str] = Field(None, description="Patient ID to look up for read or lookup action")
    text: Optional[str] = Field(None, description="User text containing an ID, used with lookup.")
    name: Optional[str] = Field(None, description="Patient's full name")
    age: Optional[int] = Field(None, description="Patient's age")
    gender: Optional[str] = Field(None, description="Patient's gender")
    contact: Optional[str] = Field(None, description="Patient's contact number")
    medical_history: Optional[str] = Field(None, description="Patient's medical history")


class CSVReadWriteTool(BaseTool):
    name: str = "CSVReadWriteTool"
    description: str = "Read patient records from a CSV file or append a new patient record with an incremental patient_id."
    args_schema: Type[BaseModel] = CSVToolInput

    def _run(
        self,
        path: str,
        action: str,
        patient_id: str = None,
        text: str = None,
        name: str = None,
        age: int = None,
        gender: str = None,
        contact: str = None,
        medical_history: str = None,
    ) -> str:
        headers = ['patient_id', 'name', 'age', 'gender', 'medical_history', 'contact']

        if action == 'read':
            records = self._load_records(path)
            if isinstance(records, str):
                return records
            if patient_id:
                return self._lookup_patient(records, patient_id)
            return json.dumps(records)

        if action == 'lookup':
            records = self._load_records(path)
            if isinstance(records, str):
                return records
            lookup_id = None
            if patient_id:
                lookup_id = str(patient_id).strip()
            elif text:
                lookup_id = self._extract_id_from_text(text)
            if not lookup_id:
                return "ERROR: No patient ID found to look up."
            return self._lookup_patient(records, lookup_id)

        if action == 'append':
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            missing = []
            if not name:
                missing.append('name')
            if age is None:
                missing.append('age')
            if not gender:
                missing.append('gender')
            if not contact:
                missing.append('contact')
            if missing:
                return f"ERROR: Missing required fields: {', '.join(missing)}"

            self._ensure_file_exists(path)
            records = self._load_records(path)
            if isinstance(records, str):
                records = []

            with open(path, mode='r+', newline='', encoding='utf-8') as fh:
                existing = list(csv.DictReader(fh))
                next_id = 1
                if existing:
                    numeric_ids = [int(row.get('patient_id', 0)) for row in existing if str(row.get('patient_id', '')).isdigit()]
                    next_id = max(numeric_ids) + 1 if numeric_ids else 1

                fh.seek(0, os.SEEK_END)
                writer = csv.DictWriter(fh, fieldnames=headers)
                if not existing:
                    writer.writeheader()
                writer.writerow({
                    'patient_id': str(next_id),
                    'name': name,
                    'age': age,
                    'gender': gender,
                    'medical_history': medical_history or '',
                    'contact': contact,
                })
                fh.flush()

            return f"SUCCESS:REGISTERED:{next_id}"

        return f"ERROR: Unsupported action '{action}' for CSVReadWriteTool."

    def _load_records(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return "ERROR: File is empty or does not exist yet."

        with open(path, mode='r', encoding='utf-8-sig') as file:
            return list(csv.DictReader(file))

    def _ensure_file_exists(self, path: str) -> None:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        open(path, 'a', encoding='utf-8').close()

    def _lookup_patient(self, reader, patient_id: str) -> str:
        patient_id_str = str(patient_id).strip()
        for row in reader:
            if str(row.get('patient_id', '')).strip() == patient_id_str:
                return json.dumps(row)
        return f"ERROR: No patient record found with patient_id={patient_id_str}."

    def _extract_id_from_text(self, text: str) -> Optional[str]:
        text = str(text)
        id_match = re.search(r'\b(?:id|ID)\b.*?(\d+)\b', text)
        if id_match:
            return id_match.group(1)
        if text.strip().isdigit():
            return text.strip()
        return None
