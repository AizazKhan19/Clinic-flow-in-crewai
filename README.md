# Clinic Flow (CrewAI) - README

## Project Overview

Clinic Flow is an interactive, agent-driven simulation of a clinical triage and treatment workflow built on top of CrewAI. It demonstrates a multi-agent conversation loop that:

- Registers patients (reads/writes a CSV "database").
- Collects and triages symptoms via a receptionist (desk) agent.
- Routes patients to specialist doctor agents (`cardiologist`, `orthopedic`, `general`).
- If necessary, simulates laboratory tests with a `Laboratory Agent` and returns a generated report.
- Lets specialist agents review lab reports and produce prescriptions or treatment advice.

This repository is intended as a proof-of-concept for building conversational clinic flows where agents coordinate, use tools, and update shared state.

## Problem It Solves

- Automates triage and basic diagnostic workflows for simulated clinical scenarios.
- Shows how to orchestrate multiple LLM-powered agents with tool integrations (CSV DB) and stateful routing.
- Provides a reproducible environment for experimenting with agent responsibilities, routing, and simulated test reporting.

## What This Provides

- A runnable Python script `clinic.py` that kicks off an interactive CLI flow.
- A small CSV-backed patient database at `data/patients_data.csv` used for registration and lookups.
- A custom CSV read/write tool implemented in `custom_tool.py` for persisting patient records.
- LLM configuration in `configuration.py` (reads `OPENAI_API_KEY` from environment).

## Repository Structure

- `clinic.py` — Main flow implementation. Defines Pydantic state models, agent tasks, routers/listeners, and the entire conversation loop.
- `configuration.py` — LLM client configuration (set `OPENAI_API_KEY` via `.env` or environment).
- `custom_tool.py` — `CSVReadWriteTool` used to read/append patient records in `data/patients_data.csv`.
- `data/patients_data.csv` — Example CSV database (header: `patient_id,name,age,gender,medical_history,contact`).
- `README.md` — This file.

## Key Components (quick reference)

- `MyClinicStates` (in `clinic.py`): global, persisted state shared across agents (patient data, flags, routing keys, `report`, etc.).
- `ClinicExtractionSchema`: the expected turn-level extraction schema for the reception desk agent.
- `DoctorDecisionState` and `LabReportState`: Pydantic schemas for doctor/lab task outputs.
- Agents: `Desk Agent` (reception), `Cardiologist/Orthopedic/General` (doctors), `Laboratory Agent` (simulates tests).
- `CSVReadWriteTool._run(...)` (in `custom_tool.py`): supports `read` and `append` actions; `append` returns a generated patient ID string.

## Requirements

- Python 3.10+ (recommended)
- Packages (example): `crewai`, `pydantic`, `python-dotenv`, `crewai-tools` (if used), and any dependencies those libraries require.

Create a minimal `requirements.txt` with:

```
pydantic
python-dotenv
crewai
crewai-tools
```

Note: exact package names and versions depend on your CrewAI installation. If you installed CrewAI via a private package, make sure to use the same environment used to develop this project.

## Configuration

1. Create a `.env` file or set environment variables with your OpenAI (or supported LLM) API key:

```
OPENAI_API_KEY=your_api_key_here
```

2. `configuration.py` reads `OPENAI_API_KEY` and initializes `llm`. Adjust model/temperature as needed.

## How to Run

From the repository root, run:

```bash
python clinic.py
```

You will interact with the desk agent via standard input. The flow continues until a specialist route completes and the flow exits.

Notes:
- `flow.plot()` is called at the end of `clinic.py`. If graph plotting requires extra dependencies (e.g., `graphviz`) you may need to install them or comment out the call.

## Data Format (`data/patients_data.csv`)

Header: `patient_id,name,age,gender,medical_history,contact`

Example row:

```
1,Aizaz Khan,45,Male,High Blood Pressure,300123456789
```

The `CSVReadWriteTool.append` action automatically generates incremental `patient_id` values.

## Example Interaction Flow (summary)

1. User provides an ID or registers as a new patient at the desk agent.
2. Desk agent extracts registration and symptoms, updates `MyClinicStates`.
3. If symptoms indicate critical condition, desk sets `chosen_doctor_path` and the router sends the state to the appropriate doctor node.
4. Doctor agent either (A) issues a prescription directly (non-critical), or (B) sets `required_tests` and the flow routes to the `lab` node.
5. `Laboratory Agent` simulates `lab_report` and sets report state; flow returns to the originating doctor node for review.

## Known Behavior & Debugging Tips

- If the doctor returns from the lab but does not add remarks:
  - Ensure that `self.state.report` is being set by the lab (search for `self.state.report = result.lab_report` in `clinic.py`).
  - Confirm the doctor node's `if self.state.report:` branch is reached; add print/log statements to verify.
  - Doctor agents produce results via `Crew(...).kickoff()`. Inspect what the crew returns and whether the response is assigned into `self.state.prescription` or another appropriate field.
  - Timing/race conditions: if lab writes happen asynchronously, make sure `is_report_generated` becomes True before routing back.
  - Check agent expected output types (`output_pydantic`) and whether the agent actually returns structured data versus free text.

## Extending the Project

- Add new specialist nodes by implementing a new `@listen(...)` method and updating the routing logic.
- Replace the simple CSV storage with a real database (SQLite/Postgres) and update `CSVReadWriteTool` or add a new DB tool.
- Replace the LLM model/config with a different provider or different parameters in `configuration.py`.
- Add unit tests that mock Crew responses to validate routing and state updates.

## Security & Privacy

- This repository simulates patient data — do NOT use real PHI/PII in test runs unless you have appropriate security controls and consent.
- Keep API keys out of source control and use environment variables or secret managers.

## Troubleshooting Checklist

- Missing API key: make sure `OPENAI_API_KEY` is set.
- CSV write/read errors: ensure `data/` directory is writable.
- Unexpected routing: inspect `self.state` logs to see which keys are set (`doctor_type`, `required_tests`, `report`).

## Contribution and License

Contributions welcome — open an issue or a pull request. This project contains example code and is intended for educational/demo use. Add an appropriate license file if you plan to publish/redistribute.

---

If you'd like, I can also:

- Add a `requirements.txt` and `.env.example`.
- Add a small `run_demo.sh` script to set env and run the flow.
- Create unit tests that mock Crew responses for deterministic behavior.

Tell me which of these you'd like next.