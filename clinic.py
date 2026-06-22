from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, router, listen, or_
from pydantic import BaseModel, ConfigDict, Field
from configuration import llm
from typing import Optional, Literal
from custom_tool import CSVReadWriteTool
from crewai.mcp import MCPServerStdio
import os
import dotenv

dotenv.load_dotenv()

# Slack server initialization
slack_server = MCPServerStdio(
    command='npx',
    args=['-y', '@modelcontextprotocol/server-slack'],
    env={
        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
        "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID")
    }
)

# Tool initialization
csv_tool = CSVReadWriteTool()

class MyClinicStates(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    # 1. Patient Core Data
    patient_id : str = None
    name : str = ""
    age : int = None
    gender : str = ""
    medical_history : str = ""
    contact : int = None
    
    # 2. Control Flags
    is_registered : bool = False      
    current_symptoms : str = ""       
    symptoms_complete : bool = False  
    doctor_type : str = ""            

    # 3. Medical & Decision Data
    prescription : str = ""            
    required_tests : str = ""          
    is_critical : bool = False         
    slack_channel_id: str = "C0BB9PTSJPR" 
    report : str = ""


# Output Schema for Info Desk
class ClinicExtractionSchema(BaseModel):
    extracted_id: Optional[str] = None
    extracted_name: Optional[str] = None
    extracted_age: Optional[int] = None
    extracted_gender: Optional[str] = None
    extracted_contact: Optional[int] = None
    extracted_history: Optional[str] = None
    extracted_symptoms: Optional[str] = None
    is_registered_now: bool = False
    symptoms_finished: bool = False
    chosen_doctor_path: str = "" 
    agent_reply: str = ""


# Schema for Doctor Decisions
class DoctorDecisionState(BaseModel):
    is_critical: bool = Field(..., description="Set to True if symptoms are dangerous/vague, False if normal.")
    prescription_or_diet: Optional[str] = Field(None, description="Provide medicines and diet plan if is_critical is False.")
    suggested_tests: Optional[str] = Field(None, description="Provide specific tests if is_critical is True.")
    doctor_reply: str = Field(..., description="Direct comforting message to display to the patient.")


# Schema for Lab Simulation
class LabReportState(BaseModel):
    is_report_generated : bool = Field(..., description="Set to True when test report is simulated, False if not.")
    lab_report : str = Field(..., description="Provide simulated lab report text based on required tests.")


class ClinicFlow(Flow[MyClinicStates]):

    @start()
    def greet_and_validation(self):
        print("============================================================")
        print(" WELCOME TO THE AI CLINIC INFORMATION DESK ")
        print("============================================================\n")
        print("Desk Agent: Welcome to Our Clinic! Please provide an ID or let me know if you are new to proceed.\n")
        
        chat_history = []
        while not self.state.symptoms_complete:
            user_msg = str(input("User Input > "))
            chat_history.append(f"User: {user_msg}")

            desk_agent = Agent(
                role="Information Desk Agent",
                goal="Manage patient check-in, registration slot-filling, and symptom triage smoothly via conversation.",
                backstory="""You are a friendly receptionist. You use tools to find IDs. If a patient is new, 
                you extract their details throughout the chat until registration is fulfilled.""",
                llm=llm,
                tools=[csv_tool],
                verbose=False
            )

            desk_task = Task(
                description=f"""You are the Clinic Information Desk Agent. Your objective is to manage patient check-in, registration validation, and symptom collection seamlessly via conversation.

                ### Input Context:
                - Current Chat History: {chat_history}
                - Latest User Message: {user_msg}
                - Current System State: {self.state.model_dump()}

                ### Workflow Logic (Follow Sequentially):
                
                1. IF USER PROVIDES AN ID:
                   - Immediately check the ID using the available tool.
                   - Do NOT output filler replies while processing.
                   - If a match is found in the database: Sync the data, set `is_registered_now` to True, greet the user warmly by their actual Name, and skip directly to Step 3 (SYMPTOMS).

                2. IF USER IS NEW (STRICT REGISTRATION):
                   - **Mandatory Fields Check:** Check the system state for Name, Age, Gender, and Contact.
                   - Do NOT mark `is_registered_now` as True if ANY of these 4 fields are missing. Keep asking for them politely.
                   - `medical_history` is OPTIONAL. If it's missing but the 4 mandatory fields are present, set `is_registered_now` to True.

                3. SYMPTOMS COLLECTION & DYNAMIC ROUTING:
                   - This step triggers only AFTER registration/check-in is complete (`is_registered_now` is True).
                   - Ask the patient about their symptoms.
                   - Once symptoms are clearly explained, set `symptoms_finished` to True.
                   - **ROUTING RULE:** Set `chosen_doctor_path` dynamically:
                     * 'cardiologist' -> If symptoms involve chest pain, heart issues, palpitations, or left arm pain radiating from chest.
                     * 'orthopedic' -> If symptoms involve bone fractures, joint pain, knee/arm injuries, swelling, or trauma.
                     * 'general' -> For all other standard, mild, or non-specific symptoms.

                ### Strict Output Guidelines:
                - Maintain a professional and comforting medical tone.
                - Ensure the `agent_reply` field contains the direct conversational response for the patient.
                """,
                expected_output="A structured Pydantic extraction containing updated registration flags, patient details, routing path, and the conversational agent reply.",
                agent=desk_agent,
                output_pydantic=ClinicExtractionSchema
            )

            turn_result = Crew(agents=[desk_agent], tasks=[desk_task], verbose=False).kickoff().pydantic

            print(f"\nDesk Agent: {turn_result.agent_reply}\n")
            chat_history.append(f"Agent: {turn_result.agent_reply}")

            if turn_result.extracted_id: self.state.patient_id = str(turn_result.extracted_id)
            if turn_result.extracted_name: self.state.name = turn_result.extracted_name
            if turn_result.extracted_age: self.state.age = turn_result.extracted_age
            if turn_result.extracted_gender: self.state.gender = turn_result.extracted_gender
            if turn_result.extracted_contact: self.state.contact = turn_result.extracted_contact
            if turn_result.extracted_history: self.state.medical_history = turn_result.extracted_history
            if turn_result.extracted_symptoms: self.state.current_symptoms = turn_result.extracted_symptoms

            if turn_result.is_registered_now and not self.state.is_registered:
                self.state.is_registered = True
                if not self.state.patient_id:
                    tool_result = csv_tool._run(
                        path='data/patients_data.csv', action="append", name=self.state.name,
                        age=self.state.age, gender=self.state.gender, contact=str(self.state.contact) if self.state.contact else "",
                        medical_history=self.state.medical_history
                    )
                    if "ID:" in tool_result:
                        self.state.patient_id = tool_result.split("ID:")[-1].strip()
                    print(f"[SYSTEM]: New Patient successfully saved to Database! ID: {self.state.patient_id}\n")

            if turn_result.symptoms_finished:
                self.state.symptoms_complete = True
                self.state.doctor_type = turn_result.chosen_doctor_path
    

    @router(greet_and_validation)
    def route_to_specialist(self) -> Literal["cardiologist", "orthopedic", "general"]:
        if self.state.doctor_type == "cardiologist":
            return "cardiologist"
        elif self.state.doctor_type == "orthopedic":
            return "orthopedic"
        else:
            return "general"

    
    @listen("cardiologist")   
    def cardiologist_node(self):
        cardio_agent = Agent(
            role='Cardiologist Agent',
            goal="Analyze symptoms, consult senior consultants via Slack for critical tracks, or perform final report analysis.",
            backstory="Veteran Cardiologist with 30+ years of experience.",
            llm=llm,
            mcps=[slack_server],
            verbose=False
        )

        if self.state.report:
            print("\n--- WELCOME BACK TO THE CARDIOLOGY DEPARTMENT (REPORT REVIEW) ---\n")
            print(f"[SYSTEM]: Dr. Heart Agent is now reviewing your Lab Report...\n")
            
            cardio_task = Task(
                description=f"""Review this simulated lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final medicine prescription, dosage, and strict cardiac diet plan.""",
                expected_output="Final clinical medicine prescription and diet layout based on lab results.",
                agent=cardio_agent
            )
            self.state.prescription = str(Crew(agents=[cardio_agent], tasks=[cardio_task], verbose=False).kickoff())
            print(f"[CARDIOLOGIST FINAL TREATMENT PLAN]:\n{self.state.prescription}\n")
            
        else:
            print("\n--- WELCOME TO THE CARDIOLOGY DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Heart Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            cardio_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Age: {self.state.age}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate serious distress, severe chest pain, or high cardiac risks, set 'is_critical' to True.
                2. If symptoms are normal or mild, set 'is_critical' to False and provide routine advice in 'prescription_or_diet'.""",
                expected_output="Structured critical assessment using DoctorDecisionState.",
                agent=cardio_agent,
                output_pydantic=DoctorDecisionState
            )

            result = Crew(agents=[cardio_agent], tasks=[cardio_task], verbose=False).kickoff().pydantic

            if result.is_critical:
                print(f"\n[SYSTEM]: Critical symptoms detected locally! Engaging Senior Consultant via Slack...")
                self.state.is_critical = True
                
                post_task = Task(
                    description=f"""Use the Slack tool to post a message into the channel ID '{self.state.slack_channel_id}'.
                    The message body MUST be exactly:
                    'Patient {self.state.name}, Age: {self.state.age} has these symptoms: {self.state.current_symptoms}. Please guide sir.'""",
                    expected_output="Confirmation that message was successfully sent.",
                    agent=cardio_agent
                )
                Crew(agents=[cardio_agent], tasks=[post_task], verbose=False).kickoff()
                
                input(f"\n[PAUSE] Alert sent to Slack. Please go to Slack, reply to the message with recommended tests, then press ENTER here to fetch guidance...")
                
                read_task = Task(
                    description=f"""Use Slack tools to fetch messages/history from channel ID '{self.state.slack_channel_id}'.
                    Find the absolute latest message or thread reply sent by the senior doctor for Patient {self.state.name}.
                    
                    STRICT HISTORY FILTER: 
                    Ignore all historical messages or previous test runs. The senior doctor's recommendation MUST match the patient's current symptoms ({self.state.current_symptoms}). If the latest message mentions cardiac tests but the patient has orthopedic issues, do NOT fetch it.""",
                    expected_output="The raw reply text from the senior doctor on slack channel.",
                    agent=cardio_agent
                )
                slack_reply = str(Crew(agents=[cardio_agent], tasks=[read_task], verbose=False).kickoff())
                print(f"\n[SYSTEM]: Raw reply fetched from Slack:\n{slack_reply}\n")
                
                analyze_task = Task(
                    description=f"""Analyze the senior doctor's Slack reply: '{slack_reply}'
                    
                    STRICT RULES:
                    1. Extract the suggested laboratory/imaging tests mentioned by the senior doctor and put them in 'suggested_tests'.
                    2. Mark 'is_critical' as True since tests have been ordered.
                    3. Fill 'doctor_reply' with a summary text updating the patient.""",
                    expected_output="Structured output mapping parameters to DoctorDecisionState.",
                    agent=cardio_agent,
                    output_pydantic=DoctorDecisionState
                )
                slack_result = Crew(agents=[cardio_agent], tasks=[analyze_task], verbose=False).kickoff().pydantic
                
                self.state.required_tests = slack_result.suggested_tests
                print(f'Cardiologist Doctor Reply (via Senior Doctor) > {slack_result.doctor_reply}')
                print(f'[SYSTEM]: Tests recommended by Senior Doctor forwarded to Lab: {self.state.required_tests}\n')
                
            else:
                self.state.is_critical = False
                self.state.prescription = result.prescription_or_diet
                self.state.required_tests = ""  
                print(f'Cardiologist Doctor Reply > {result.doctor_reply}\n')
                print(f'[SYSTEM]: Mild symptoms track completed. Directing straight to flow termination.\n')


    @listen("orthopedic")
    def orthopedic_node(self):
        ortho_agent = Agent(
            role='Orthopedic Agent',
            goal="Analyze musculoskeletal symptoms, consult senior consultants via Slack for trauma/fractures, or review imaging reports.",
            backstory="Veteran Orthopedic Surgeon with 30+ years of experience.",
            llm=llm,
            mcps=[slack_server],
            verbose=False
        )

        if self.state.report:
            print("\n--- WELCOME BACK TO THE ORTHOPEDIC DEPARTMENT (REPORT REVIEW) ---\n")
            print(f"[SYSTEM]: Dr. Bone Agent is now reviewing your Lab/Imaging Report...\n")
            
            ortho_task = Task(
                description=f"""Review this simulated imaging/lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final orthopedic medicine prescription, immobilization advice, and a bone recovery nutritional layout.""",
                expected_output="Final bone clinical prescription and recovery layout based on lab results.",
                agent=ortho_agent
            )
            self.state.prescription = str(Crew(agents=[ortho_agent], tasks=[ortho_task], verbose=False).kickoff())
            print(f"[ORTHOPEDIC FINAL TREATMENT PLAN]:\n{self.state.prescription}\n")
            
        else:
            print("\n--- WELCOME TO THE ORTHOPEDIC DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Bone Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            ortho_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Age: {self.state.age}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate acute physical trauma, severe swelling, suspected fractures, or joint dislocation, set 'is_critical' to True to order X-Rays/MRIs.
                2. If symptoms are minor strains or mild sprains, set 'is_critical' to False and provide routine care instructions.""",
                expected_output="Structured orthopedic assessment using DoctorDecisionState.",
                agent=ortho_agent,
                output_pydantic=DoctorDecisionState
            )

            result = Crew(agents=[ortho_agent], tasks=[ortho_task], verbose=False).kickoff().pydantic

            if result.is_critical:
                print(f"\n[SYSTEM]: Critical bone/joint symptoms detected! Engaging Senior Orthopedic Consultant via Slack...")
                self.state.is_critical = True
                
                post_task = Task(
                    description=f"""Use the Slack tool to post a message into the channel ID '{self.state.slack_channel_id}'.
                    The message body MUST be exactly:
                    'Patient {self.state.name}, Age: {self.state.age} has these symptoms: {self.state.current_symptoms}. Please guide sir.'""",
                    expected_output="Confirmation that message was successfully sent.",
                    agent=ortho_agent
                )
                Crew(agents=[ortho_agent], tasks=[post_task], verbose=False).kickoff()
                
                input(f"\n[PAUSE] Alert sent to Slack. Please go to Slack, reply to the message with recommended tests, then press ENTER here to fetch guidance...")
                
                read_task = Task(
                    description=f"""Use Slack tools to fetch messages/history from channel ID '{self.state.slack_channel_id}'.
                    Find the absolute latest message or thread reply sent by the senior doctor for Patient {self.state.name}.
                    
                    STRICT HISTORY FILTER: 
                    Ignore all historical messages or previous test runs. The senior doctor's recommendation MUST match the patient's current orthopedic symptoms ({self.state.current_symptoms}). Completely ignore cardiac test suggestions.""",
                    expected_output="The raw reply text from the senior doctor on slack channel.",
                    agent=ortho_agent
                )
                slack_reply = str(Crew(agents=[ortho_agent], tasks=[read_task], verbose=False).kickoff())
                print(f"\n[SYSTEM]: Raw reply fetched from Slack:\n{slack_reply}\n")
                
                analyze_task = Task(
                    description=f"""Analyze the senior doctor's Slack reply: '{slack_reply}'
                    
                    STRICT RULES:
                    1. Extract the suggested imaging/laboratory tests (e.g., X-Ray, MRI) mentioned by the senior doctor and put them in 'suggested_tests'.
                    2. Mark 'is_critical' as True.
                    3. Fill 'doctor_reply' with a summary text updating the patient.""",
                    expected_output="Structured output mapping parameters to DoctorDecisionState.",
                    agent=ortho_agent,
                    output_pydantic=DoctorDecisionState
                )
                slack_result = Crew(agents=[ortho_agent], tasks=[analyze_task], verbose=False).kickoff().pydantic
                
                self.state.required_tests = slack_result.suggested_tests
                print(f'Orthopedic Doctor Reply (via Senior Doctor) > {slack_result.doctor_reply}')
                print(f'[SYSTEM]: Tests recommended by Senior Doctor forwarded to Lab: {self.state.required_tests}\n')
                
            else:
                self.state.is_critical = False
                self.state.prescription = result.prescription_or_diet
                self.state.required_tests = ""  
                print(f'Orthopedic Doctor Reply > {result.doctor_reply}\n')
                print(f'[SYSTEM]: Mild orthopedic track completed. Directing straight to flow termination.\n')


    @listen("general")
    def general_node(self):
        general_agent = Agent(
            role='General Physician Agent',
            goal="Analyze general medical symptoms, consult senior consultants via Slack if needed, or perform final review.",
            backstory="Veteran General Physician with 30+ years of experience.",
            llm=llm,
            mcps=[slack_server],
            verbose=False
        )

        if self.state.report:
            print("\n--- WELCOME BACK TO THE GENERAL DEPARTMENT (REPORT REVIEW) ---\n")
            print(f"[SYSTEM]: Dr. General Agent is now reviewing your Lab Report...\n")
            
            general_task = Task(
                description=f"""Review this simulated lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final medicine prescription, dosage, and general wellness diet plan.""",
                expected_output="Final clinical medicine prescription and diet layout based on lab results.",
                agent=general_agent
            )
            self.state.prescription = str(Crew(agents=[general_agent], tasks=[general_task], verbose=False).kickoff())
            print(f"[GENERAL PHYSICIAN FINAL TREATMENT PLAN]:\n{self.state.prescription}\n")
            
        else:
            print("\n--- WELCOME TO THE GENERAL DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. General Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            general_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Age: {self.state.age}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate an unidentifiable systemic issue or critical vital instabilities, set 'is_critical' to True.
                2. If symptoms are standard or mild, set 'is_critical' to False and provide routine advice.""",
                expected_output="Structured critical assessment using DoctorDecisionState.",
                agent=general_agent,
                output_pydantic=DoctorDecisionState
            )

            result = Crew(agents=[general_agent], tasks=[general_task], verbose=False).kickoff().pydantic

            if result.is_critical:
                print(f"\n[SYSTEM]: Critical general symptoms detected locally! Engaging Senior Consultant via Slack...")
                self.state.is_critical = True
                
                post_task = Task(
                    description=f"""Use the Slack tool to post a message into the channel ID '{self.state.slack_channel_id}'.
                    The message body MUST be exactly:
                    'Patient {self.state.name}, Age: {self.state.age} has these symptoms: {self.state.current_symptoms}. Please guide sir.'""",
                    expected_output="Confirmation that message was successfully sent.",
                    agent=general_agent
                )
                Crew(agents=[general_agent], tasks=[post_task], verbose=False).kickoff()
                
                input(f"\n[PAUSE] Alert sent to Slack. Please go to Slack, reply to the message with recommended tests, then press ENTER here to fetch guidance...")
                
                read_task = Task(
                    description=f"""Use Slack tools to fetch messages/history from channel ID '{self.state.slack_channel_id}'.
                    Find the absolute latest response text reply sent by the senior doctor for Patient {self.state.name}.
                    
                    STRICT HISTORY FILTER:
                    Ignore all historical records or previous diagnostic iterations. Must match current symptoms ({self.state.current_symptoms}).""",
                    expected_output="The raw reply text from the senior doctor on slack channel.",
                    agent=general_agent
                )
                slack_reply = str(Crew(agents=[general_agent], tasks=[read_task], verbose=False).kickoff())
                print(f"\n[SYSTEM]: Raw reply fetched from Slack:\n{slack_reply}\n")
                
                analyze_task = Task(
                    description=f"""Analyze the senior doctor's Slack reply: '{slack_reply}'
                    
                    STRICT RULES:
                    1. Extract the suggested laboratory tests mentioned by the senior doctor and put them in 'suggested_tests'.
                    2. Mark 'is_critical' as True.
                    3. Fill 'doctor_reply' with a summary text updating the patient.""",
                    expected_output="Structured output mapping parameters to DoctorDecisionState.",
                    agent=general_agent,
                    output_pydantic=DoctorDecisionState
                )
                slack_result = Crew(agents=[general_agent], tasks=[analyze_task], verbose=False).kickoff().pydantic
                
                self.state.required_tests = slack_result.suggested_tests
                print(f'GENERAL Doctor Reply (via Senior Doctor) > {slack_result.doctor_reply}')
                print(f'[SYSTEM]: Tests recommended by Senior Doctor forwarded to Lab: {self.state.required_tests}\n')
                
            else:
                self.state.is_critical = False
                self.state.prescription = result.prescription_or_diet
                self.state.required_tests = ""  
                print(f'GENERAL Doctor Reply > {result.doctor_reply}\n')
                print(f'[SYSTEM]: Mild symptoms track completed. Directing straight to flow termination.\n')


    # Laboratory Router & Execution Node
    @router(or_(cardiologist_node, orthopedic_node, general_node))
    def lab(self) -> Literal["cardiologist", "orthopedic", "general", "exit_flow"]:
        if self.state.required_tests and not self.state.report:
            print(f'[SYSTEM]: Performing prescribed tests in Laboratory: {self.state.required_tests} \n')
            
            lab_agent = Agent(
                role="Laboratory Agent",
                goal="Simulate realistic lab results based on tests requested by the treating physician.",
                backstory="Expert Laboratory Technician responsible for building structured mock clinical reports.",
                llm=llm, verbose=False
            )

            lab_task = Task(
                description=f"""Simulate a professional clinical report for these tests: {self.state.required_tests}.
                Use the following demographic context matching current system states:
                Name: {self.state.name}
                Age: {self.state.age}
                Gender: {self.state.gender}""",
                expected_output="Realistic structured lab report mapping metric values.",
                agent=lab_agent, 
                output_pydantic=LabReportState
            )

            result = Crew(agents=[lab_agent], tasks=[lab_task], verbose=False).kickoff().pydantic

            if result.is_report_generated:
                self.state.report = result.lab_report
                print(f'[SYSTEM]: Lab report generated! Routing back to department for review...\n')
                
                if self.state.doctor_type == "cardiologist":
                    return "cardiologist"
                elif self.state.doctor_type == "orthopedic": 
                    return "orthopedic"
                else:
                    return "general"
        else:
            return "exit_flow"
        
    @listen("exit_flow")
    def flow_exit(self):
       
        print(f' THE CLINIC FLOW IS TERMINATED SUCCESSFULLY ')
        
        

flow = ClinicFlow()
flow.kickoff()
flow.plot()