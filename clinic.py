from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, router, listen, or_
from pydantic import BaseModel, ConfigDict, Field
from crewai.tools import BaseTool
from configuration import llm
from crewai_tools import CSVSearchTool
from typing import Optional, Literal, Type
from custom_tool import CSVReadWriteTool

# tool's object initialization
csv_tool = CSVReadWriteTool()

class MyClinicStates(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    # 1. Patient Core Data (Registration ke liye)
    patient_id : str = None
    name : str = ""
    age : int = None
    gender : str = ""
    medical_history : str = ""
    contact : int = None
    
    # 2. Control Flags (Loop chalane aur check karne ke liye)
    is_registered : bool = False      # True hoga jab naya patient register ho jaye ya purana mil jaye
    current_symptoms : str = ""       # User ke bataye hue saare symptoms yahan jama honge
    symptoms_complete : bool = False  # Agent isko True karega jab mazeed follow-up ki zaroorat nahi hogi
    
    # 3. Router Key
    doctor_type : str = ""            # Isme final path string save hogi

    # Doctor Decision states global
    doctor_notes : str = ""            # Doctor ke remarks (Normal ya Critical)
    prescription : str = ""            # Agar normal hua toh medicine/diet yahan aayegi
    required_tests : str = ""          # Agar critical hua toh tests yahan aayenge
    is_critical : bool = False         # Laboratory routing ke liye flag
    doctor : str = ""                  # the name of doctor who suggested tests

    

#  After every iteration , Agent will output this  kind of dat
class ClinicExtractionSchema(BaseModel):
    # Registration Extraction
    extracted_id: Optional[str] = None
    extracted_name: Optional[str] = None
    extracted_age: Optional[int] = None
    extracted_gender: Optional[str] = None
    extracted_contact: Optional[int] = None
    extracted_history: Optional[str] = None

    extracted_symptoms: Optional[str] = None
    
    # Flags & Triage
    is_registered_now: bool = False
    symptoms_finished: bool = False
    chosen_doctor_path: str = "" # 'cardiologist_path', 'orthopedic_path', 'general_path'
    
    # The actual response to display
    agent_reply: str = ""

    

# doctor decision state which agent will use to update global states
class DoctorDecisionState(BaseModel):
    is_critical: bool = Field(..., description="Set to True if symptoms are dangerous/vague, False if normal.")
    prescription_or_diet: Optional[str] = Field(None, description="Provide medicines and diet plan if is_critical is False.")
    suggested_tests: Optional[str] = Field(None, description="Provide specific tests (like ECG, CBC) if is_critical is True.")
    doctor_reply: str = Field(..., description="The direct comforting message to display to the patient.")




class ClinicFlow(Flow[MyClinicStates]):

    @start()

    def greet_and_validation(self):
        print("============================================================")
        print(" WELCOME TO THE AI CLINIC INFORMATION DESK ")
        print("============================================================\n")
        
        print("Desk Agent: Welcome to Our Clinic! Please provide an ID or let me know if you are new to proceed.\n")
        
        chat_history = []
        csv_tool = CSVReadWriteTool()

        # Main Loop: Jab tak doctor route nahi milta chat chalti rahegi
        while not self.state.symptoms_complete:
            user_msg = input("User Input > ")
            chat_history.append(f"User: {user_msg}")

            # 1. Agent Definition (Aap ka design)
            desk_agent = Agent(
                role="Information Desk Agent",
                goal="Manage patient check-in, registration slot-filling, and symptom triage smoothly via conversation.",
                backstory="""You are a friendly receptionist. You use tools to find IDs. If a patient is new, 
                you extract their details (name, age, gender, contact, history) throughout the chat. 
                Once registered or found, you collect symptoms until you are 100% sure of their specialist route.""",
                llm=llm,
                tools=[csv_tool],
                verbose=False
            )

            # 2. Task Definition (Aap ke prompts ko interactive banaya)
            desk_task = Task(
                description=f"""Analyze the entire chat history and the latest user response.
                
                Current Chat History: {chat_history}
                Latest User Message: {user_msg}
                Current System States: {self.state.model_dump()}

                STRICT INSTRUCTIONS:
                1. ID CHECK: If user provided an ID and 'is_registered' is False, execute CSVReadWriteTool with that ID string. Look closely at the tool output. If the tool finds a match for that specific ID row, mark 'is_registered_now' as True, extract all details from that row, and map the ID to 'extracted_id'.
                2. REGISTRATION: If ID not found or tool returns empty/no match, ask for Name, Age, Gender, Contact, medical history (optional/  if have). Extract whatever they provide in the latest message. If all provided, set 'is_registered_now' to True. else ask for remaining things and then when you got all the things then set 'is_registered_now' to True.
                3. SYMPTOMS: If patient is registered/found, ask for symptoms. Analyze carefully. If vague, ask follow-ups. If 100% certain, set 'symptoms_finished' to True and 'chosen_doctor_path' to one of: 'cardiologist_path', 'orthopedic_path', 'general_path'.
                4. BOUNDARY: Stay strictly within clinic domain. Politely refuse politics/jokes etc.
                5. Provide your next natural response in 'agent_reply'.""",
                expected_output="Structured chat turn analysis and data extraction.",
                agent=desk_agent,
                output_pydantic=ClinicExtractionSchema
            )

            crew = Crew(agents=[desk_agent], tasks=[desk_task], verbose=False)
            turn_result = crew.kickoff().pydantic

            # --- Python Side State Synchronization ---
            print(f"\nDesk Agent: {turn_result.agent_reply}\n")
            chat_history.append(f"Agent: {turn_result.agent_reply}")

            if turn_result.extracted_id: 
                self.state.patient_id = str(turn_result.extracted_id)

            # State Updates based on extraction
            if turn_result.extracted_name: self.state.name = turn_result.extracted_name
            if turn_result.extracted_age: self.state.age = turn_result.extracted_age
            if turn_result.extracted_gender: self.state.gender = turn_result.extracted_gender
            if turn_result.extracted_contact: self.state.contact = turn_result.extracted_contact
            if turn_result.extracted_history: self.state.medical_history = turn_result.extracted_history
            if turn_result.extracted_symptoms: 
                self.state.current_symptoms = turn_result.extracted_symptoms

            # Check if registration just completed
            if turn_result.is_registered_now and not self.state.is_registered:
                self.state.is_registered = True
                
                # Agar state mein pehle se patient_id nahi hai (yaani naya user hai)
                if not self.state.patient_id:
                    # 1. Tool ka instance use karte hue data save karo
                    csv_path = 'data/patients_data.csv'
                    tool_result = csv_tool._run(
                        path=csv_path,
                        action="append",
                        name=self.state.name,
                        age=self.state.age,
                        gender=self.state.gender,
                        contact=str(self.state.contact) if self.state.contact else "",
                        medical_history=self.state.medical_history
                    )
                    
                    # 2. Tool return karega: "Success: Registered with ID: X"
                    # Hum string se sirf number extract kar ke state mein save kar lenge
                    if "ID:" in tool_result:
                        new_id = tool_result.split("ID:")[-1].strip()
                        self.state.patient_id = new_id
                        turn_result.extracted_id = new_id # Agent ke schema ko bhi sync kar diya
                    
                    print(f"[SYSTEM]: New Patient successfully saved to CSV Database!")
                    print(f"[SYSTEM]: Generated Patient ID is: {self.state.patient_id}\n")

            # Check if triage is done
            if turn_result.symptoms_finished:
                self.state.symptoms_complete = True
                self.state.doctor_type = turn_result.chosen_doctor_path
    


    # ------------------------------------------------------------
    # 2. ROUTER: Yeh tay karega ke flow kis doctor ke paas jayega
    # ------------------------------------------------------------
    @router(greet_and_validation)
    def route_to_specialist(self)-> Literal["cardiologist", "orthopedic", "general"]:
        if self.state.doctor_type == "cardiologist_path":
            return "cardiologist"
        elif self.state.doctor_type == "orthopedic_path":
            return "orthopedic"
        else:
            return "general"

    # ------------------------------------------------------------
    # 3. LISTENERS (DOCTOR NODES): Jin par router bhejega
    # ------------------------------------------------------------
    @listen("cardiologist")
    def cardiologist_node(self) ->Literal['test_required']:

        print("\n--- WELCOME TO THE CARDIOLOGY DEPARTMENT ---")
        print(f"[SYSTEM]: Dr. Heart Agent is analyzing symptoms for {self.state.name}...\n")

        cardio_agent = Agent(
            role='Cardiologist Agent',
            goal="To give treatment or recommend tests based on patient symptoms analysis if its realed to heart.",
            backstory="""You are a Cardiologist Doctor specialized in heart issues with 30+ years of experience. 
            You carefully analyze symptoms and decide whether the condition is 'normal' (needs medicine/diet) 
            or 'critical' (needs laboratory tests).""",
            llm=llm,
            verbose=False
        )

        cardio_task = Task(
            description=f"""Analyze the patient's context deeply:
            Patient Name: {self.state.name}
            Current Symptoms: {self.state.current_symptoms}
            Medical History: {self.state.medical_history}
            
            STRICT RULES:
            1. If symptoms indicate severe distress, radiating chest pain, or high-risk history, set 'is_critical' to True and list required labs in 'suggested_tests'.
            2. If symptoms are stable or routine, set 'is_critical' to False and provide data in 'prescription_or_diet'.
            3. Write your final direct response to the patient in 'doctor_reply'.""",
            expected_output="Structured critical assessment and medical guidance.",
            agent=cardio_agent,
            output_pydantic=DoctorDecisionState
        )

        crew = Crew(agents=[cardio_agent], tasks=[cardio_task], verbose=False)
        result = crew.kickoff().pydantic


        # states synchronization

        if result.is_critical:
            self.state.required_tests = result.suggested_tests
            print(f'Cardiologist doctor reply > {result.doctor_reply}')
            return "test_required"
        else:
            self.state.prescription = result.prescription_or_diet
            print(f' Cardiologist Doctor Prescription : {result.doctor_reply}')

        

    @listen("orthopedic")
    def orthopedic_node(self):

        print("\n--- WELCOME TO THE ORTHOPEDIC DEPARTMENT ---")
        print(f"[SYSTEM]: Dr. Bone Agent is analyzing symptoms for {self.state.name}...\n")

        ortho_agent = Agent(
            role='Orthopedic Agent',
            goal="To give treatment or recommend tests based on patient symptoms analysis if its related to bones.",
            backstory="""You are a Orthopedic Doctor specialized in Bone issues with 30+ years of experience. 
            You carefully analyze symptoms and decide whether the condition is 'normal' (needs medicine/diet) 
            or 'critical' (needs laboratory tests).""",
            llm=llm,
            verbose=False
        )

        ortho_task = Task(
            description=f"""Analyze the patient's context deeply:
            Patient Name: {self.state.name}
            Current Symptoms: {self.state.current_symptoms}
            Medical History: {self.state.medical_history}
            
            STRICT RULES:
            1. If symptoms indicate severe distress, radiating bone pain, or high-risk history, set 'is_critical' to True and list required labs in 'suggested_tests'.
            2. If symptoms are stable or routine, set 'is_critical' to False and provide data in 'prescription_or_diet'.
            3. Write your final direct response to the patient in 'doctor_reply'.""",
            expected_output="Structured critical assessment and medical guidance.",
            agent=ortho_agent,
            output_pydantic=DoctorDecisionState
        )

        crew = Crew(agents=[ortho_agent], tasks=[ortho_task], verbose=False)
        result = crew.kickoff().pydantic

        if result.is_critical:
            self.state.required_tests = result.suggested_tests
            print(f'Orthopedic doctor reply > {result.doctor_reply}')
            return "test_required"
        else:
            self.state.prescription = result.prescription_or_diet
            print(f' Orthopedic Doctors Prescription : {result.doctor_reply}')



        

    @listen("general")
    def general_node(self):
        print("\n--- WELCOME TO THE GENERAL DEPARTMENT ---")
        print(f"[SYSTEM]: Dr. General Agent is analyzing symptoms for {self.state.name}...\n")

        general_agent = Agent(
            role='General Agent',
            goal="To give treatment or recommend tests based on patient symptoms analysis if its related to general health.",
            backstory="""You are a General Physician Doctor specialized in general issues with 30+ years of experience. 
            You carefully analyze symptoms and decide whether the condition is 'normal' (needs medicine/diet) 
            or 'critical' (needs laboratory tests).""",
            llm=llm,
            verbose=False
        )

        general_task = Task(
            description=f"""Analyze the patient's context deeply:
            Patient Name: {self.state.name}
            Current Symptoms: {self.state.current_symptoms}
            Medical History: {self.state.medical_history}
            
            STRICT RULES:
            1. If symptoms indicate severe distress, or high-risk history, set 'is_critical' to True and list required labs in 'suggested_tests'.
            2. If symptoms are stable or routine, set 'is_critical' to False and provide data in 'prescription_or_diet'.
            3. Write your final direct response to the patient in 'doctor_reply'.""",
            expected_output="Structured critical assessment and medical guidance.",
            agent=general_agent,
            output_pydantic=DoctorDecisionState
        )

        crew = Crew(agents=[general_agent], tasks=[general_task], verbose=False)
        result = crew.kickoff().pydantic

        if result.is_critical:
            self.state.required_tests = result.suggested_tests
            print(f'General doctor reply > {result.doctor_reply}')
            return "test_required"
        else:
            self.state.prescription = result.prescription_or_diet
            print(f' General Doctor Prescription : {result.doctor_reply}')


    # lab fucntion
    @listen(or_(cardiologist_node, orthopedic_node, general_node))
    def lab(self):
        print(f'Performing prescribed tests in Lab : { self.state.required_tests}')
        
    


flow = ClinicFlow()

flow.kickoff()
flow.plot()
