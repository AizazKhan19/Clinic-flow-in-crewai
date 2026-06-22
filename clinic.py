from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, router, listen, or_
from pydantic import BaseModel, ConfigDict, Field
from crewai.tools import BaseTool
from configuration import llm
from crewai_tools import CSVSearchTool
from typing import Optional, Literal, Type
from custom_tool import CSVReadWriteTool
from crewai.mcp import MCPServerStdio
import os
import dotenv

dotenv.load_dotenv()

# slack server initialization

slack_server = MCPServerStdio(
    command='npx',
    args=['-y', ' @modelcontextprotocol/server-slack '],

    env={

        "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),

        "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID")

}
)

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
    # doctor_notes : str = ""            # Doctor ke remarks (Normal ya Critical)
    prescription : str = ""            # Agar normal hua toh medicine/diet yahan aayegi
    required_tests : str = ""          # Agar critical hua toh tests yahan aayenge
    is_critical : bool = False         # Laboratory routing ke liye flag
    # doctor : str = ""                  # the name of doctor who suggested tests

    # Lab testting state
    report : str = ""
    is_report_ready : bool = False   # set to True when report is simulated

    
#  After every iteration , Agent will output this  kind of data
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

class LabReportState(BaseModel):
    is_report_generated : bool = Field(..., description= "set to True when test report is simulated, False if not simulated")
    lab_report : str = Field(..., description= " Provide simulated lab report generated based on suggested tests")


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
            user_msg = str(input("User Input > "))
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

                1. ID CHECK: If user provided an ID and 'is_registered' is False, 
                execute CSVReadWriteTool with that ID string. Look closely at the tool output. 
                If the tool finds a match for that specific ID row, mark 'is_registered_now' as True, 
                extract all details from that row and save them in desired states then must ask about symptoms.
                Whenever you are matching user's provided id using tool, then do not message to user saying such as
                wait you have provided id now let me take a look if it is found or not? instead when you found it then 
                tell user that you found it by mentioning user's name which you map from already present user's
                data against id in csv file.
                For example save name in 'extracted_name', age in 'extracted_age',
                gender in 'extracted_gender', contact in 'extracted_contact' and history in 'extracted_history'.
                and map the ID to 'extracted_id'.

                2. REGISTRATION: If ID not found then:

                    i. Collect mandatory information including 'Name', 'Age', 'Gender', 'Contact',
                    medical history (optional/  if have).
                    ii. You must clearly check what user is providing, if user do not have medical history then you can proceed
                    registration without it BUT if user do not have , name, contact, age, gender or even one of these
                    things then you are not allowed to proceed registration. 
                    iii. You must proceed and registered user successfully when user you have user's name, age, gender and contact.
                    and if user do not have even one of these ( name, age, contact, gender). Then you MUST NOT PROCEED WITH 
                    REGISTRATION. 
                    iv. Extract whatever they provide in the latest message but keep in mind that what is relevant 
                    for registration (as discussed above ).
                    v. If all provided, set 'is_registered_now' to True. else ask for remaining things and then, 
                     when you got all the things then set 'is_registered_now' to True. 

                3. SYMPTOMS: If patient is registered/found :

                  i. Ask for symptoms.
                  ii. Analyze carefully. If vague, ask follow-ups questions to clearly understand symptoms. 
                  iii. If 100% clear about symptoms then, set 'symptoms_finished' to True and 'chosen_doctor_path' 
                  to one of: 'cardiologist', 'orthopedic', 'general'.

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
                    print("--- Manually appending record in csv file...")
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
    

    # ROUTER: Yeh tay karega ke flow kis doctor ke paas jayega    
    @router(greet_and_validation)
    def route_to_specialist(self)-> Literal["cardiologist", "orthopedic", "general"]:

        if self.state.doctor_type == "cardiologist":
            return "cardiologist"
        elif self.state.doctor_type == "orthopedic":
            return "orthopedic"
        else:
            return "general"

    
    # LISTENERS (DOCTOR NODES): Jin par router bhejega
    
    @listen("cardiologist")   
    def cardiologist_node(self):

        # 1. Unified Agent Definition (Handles both Initial Diagnosis and Lab Review)
        cardio_agent = Agent(
            role='Cardiologist Agent',
            goal="To analyze patient symptoms, recommend diagnostic laboratory tests if critical, or review generated lab reports to provide final clinical treatment.",
            backstory="""You are a veteran Cardiologist Doctor with 30+ years of experience.
            You excel at two main stages:
            Stage 1: Initially diagnosing patient chest pains/symptoms to see if they are critical or normal.
            Stage 2: Thoroughly inspecting technical medical lab reports once generated, mapping results to symptoms, and issuing final prescriptions.""",
            llm=llm,
            verbose=False
        )

        # Sync doctor identity globally for the lab tracking
        # self.state.doctor = 'Cardiologist Doctor'

        
        # STAGE 2: Lab Report Review Flow (Triggers if report exists)
        
        if self.state.report:
            print("\n--- WELCOME BACK TO THE CARDIOLOGY DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Heart Agent is now reviewing your Lab Report...\n")
            
            
            # Simple review task returning a raw text prescription string
            cardio_task = Task(
                description=f"""Review this simulated lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final medicine prescription, dosage, and strict cardiac diet plan.""",
                expected_output="Final clinical medicine prescription and diet layout based on lab results.",
                agent=cardio_agent
            )
            
            # Execute Review Task and save final text to state
            self.state.prescription = str(Crew(agents=[cardio_agent], tasks=[cardio_task], verbose=False).kickoff())
            print(f"[CARDIOLOGIST DOCTOR'S PRESCRIPTION AFTER ASSESSING THE REPORT]:\n{self.state.prescription}\n")
            
            

        
        # STAGE 1: Initial Symptom Diagnosis Flow (Triggers on first visit)
        
        else:

            print("\n--- WELCOME TO THE CARDIOLOGY DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Heart Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            # Structured task returning Pydantic schema for routing/extraction
            cardio_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate severe distress, radiating chest pain, or high-risk history, 
                set 'is_critical' to True and list required labs in 'suggested_tests'.
                2. If symptoms are mild or seems like minor symptoms like not dangerous, 
                set 'is_critical' to False and provide prescription to that mild symptoms in 'prescription_or_diet'.
                3. If 'is_critical' is false which means mild symptoms then :
                i. Write prescription to the patient in 'doctor_reply' using prescription which you put in 
                'prescription_or_diet'.
                
                4. If 'is_Critical' is True which means severe symptoms or symptoms seems dangerous/critical then:
                i. Write your direct response to the patient in 'doctor_reply' saying such as your symptoms seems
                critical so i suggest you some tests ( which you put in 'suggested_test). You first
                do that and then i will see your report and will give my remarks on.""",
                expected_output="Structured critical assessment and medical guidance.",
                agent=cardio_agent,
                output_pydantic=DoctorDecisionState
            )

            # Execute Initial Assessment Crew
            crew = Crew(agents=[cardio_agent], tasks=[cardio_task], verbose=False)
            result = crew.kickoff().pydantic

            # Dynamic Path Branching
            if result.is_critical:
                self.state.required_tests = result.suggested_tests
                print(f'Cardiologist Doctor Reply > {result.doctor_reply}')
                print(f'[SYSTEM]: Test ordered: {self.state.required_tests}. Routing to Lab...\n')
                
            else:
                self.state.prescription = result.prescription_or_diet
                print(f'Cardiologist Doctor Reply > {result.doctor_reply}\n')

            

        

    
    @listen("orthopedic")
    def orthopedic_node(self):

        # 1. Unified Agent Definition (Handles both Initial Assessment and Lab Review)
        ortho_agent = Agent(
            role='Orthopedic Agent',
            goal="To analyze patient bone and joint symptoms, recommend diagnostic laboratory or imaging tests if critical, or review generated lab reports to provide final clinical treatment.",
            backstory="""You are a Specialized Orthopedic Doctor with 30+ years of experience.
            You excel at two main stages:
            Stage 1: Initially diagnosing patient bone/joint trauma or pain to see if they are critical or normal.
            Stage 2: Thoroughly inspecting technical medical lab or imaging reports once generated, mapping results to symptoms, and issuing final prescriptions.""",
            llm=llm,
            verbose=False
        )

        # Sync doctor identity globally for the lab tracking
        # self.state.doctor = 'Orthopedic Doctor'

        
        # Lab Report Review Flow (Triggers if report exists)
        
        if self.state.report:
            print("\n--- WELCOME BACK TO THE ORTHOPEDIC DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Bone Agent is now reviewing your Lab Report...\n")
            
            # Simple review task returning a raw text prescription string
            ortho_task = Task(
                description=f"""Review this simulated lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final medicine prescription, physical therapy layout, and recovery advice.""",
                expected_output="Final clinical medicine prescription and bone recovery layout based on lab results.",
                agent=ortho_agent
            )
            
            # Execute Review Task and save final text to state
            self.state.prescription = str(Crew(agents=[ortho_agent], tasks=[ortho_task], verbose=False).kickoff())
            
            print(f"[ORTHOPEDIC DOCTOR'S PRESCRIPTION AFTER ASSESSING THE REPORT]:\n{self.state.prescription}\n")
            # return  # Exits the loop cleanly

        
        # Initial Symptom Diagnosis Flow (Triggers on first visit)
        
        else:
            print("\n--- WELCOME TO THE ORTHOPEDIC DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. Bone Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            # Structured task returning Pydantic schema for routing/extraction
            ortho_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate severe distress, radiating bone pain, suspected fracture, or high-risk history, 
                set 'is_critical' to True and list required scans/labs (like X-Ray, MRI, etc.) in 'suggested_tests'.
                2. If symptoms are mild or seems like minor symptoms like routine muscle strain or not dangerous, 
                set 'is_critical' to False and provide data in 'prescription_or_diet'.
                3. If 'is_critical' is false which means mild symptoms then :
                i. Write your prescription to the patient in 'doctor_reply' using prescription which you put in 
                'prescription_or_diet'.
                
                4. If 'is_critical' is True which means severe symptoms or symptoms seems dangerous/critical then:
                i. Write your direct response to the patient in 'doctor_reply' saying such as your symptoms seems
                critical so i suggest you some tests ( which you put in 'suggested_test'). You first
                do that and then i will see your report and will give my remarks on.""",
                expected_output="Structured critical assessment and medical guidance.",
                agent=ortho_agent,
                output_pydantic=DoctorDecisionState
            )

            # Execute Initial Assessment Crew
            crew = Crew(agents=[ortho_agent], tasks=[ortho_task], verbose=False)
            result = crew.kickoff().pydantic

            # Dynamic Path Branching
            if result.is_critical:
                self.state.required_tests = result.suggested_tests
                print(f'Orthopedic Doctor Reply > {result.doctor_reply}')
                print(f'[SYSTEM]: Test ordered: {self.state.required_tests}. Routing to Lab...\n')
                
            else:
                self.state.prescription = result.prescription_or_diet
                print(f'Orthopedic Doctor Reply > {result.doctor_reply}\n')


    @listen("general")
    def general_node(self):
        
        # 1. Unified Agent Definition (Handles both Initial Assessment and Lab Review)
        general_agent = Agent(
            role='General Agent',
            goal="To analyze patient general health symptoms, recommend diagnostic laboratory tests if critical, or review generated lab reports to provide final clinical treatment.",
            backstory="""You are a veteran General Physician Doctor with 30+ years of experience.
            You excel at two main stages:
            Stage 1: Initially diagnosing patient general health symptoms to see if they are critical or normal.
            Stage 2: Thoroughly inspecting technical medical lab reports once generated, mapping results to symptoms, and issuing final prescriptions.""",
            llm=llm,
            verbose=False
        )

        # Sync doctor identity globally for the lab tracking
        # self.state.doctor = 'General Physician'

        
        # Lab Report Review Flow (Triggers if report exists)
        
        if self.state.report:
            print("\n--- WELCOME BACK TO THE GENERAL DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. General Agent is now reviewing your Lab Report...\n")
            
            # Simple review task returning a raw text prescription string
            general_task = Task(
                description=f"""Review this simulated lab report deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Generated Lab Report Data: '{self.state.report}'
                
                Based on these report values, provide the final medicine prescription, dosage, and general healthcare advice.""",
                expected_output="Final clinical medicine prescription and primary care layout based on lab results.",
                agent=general_agent
            )
            
            # Execute Review Task and save final text to state
            self.state.prescription = str(Crew(agents=[general_agent], tasks=[general_task], verbose=False).kickoff())
            
            print(f"[GENERAL DOCTOR'S PRESCRIPTION AFTER ASSESSING THE REPORT]:\n{self.state.prescription}\n")
            # return  # Exits the loop cleanly

        
        # Initial Symptom Diagnosis Flow (Triggers on first visit)
        
        else:
            print("\n--- WELCOME TO THE GENERAL DEPARTMENT ---\n")
            print(f"[SYSTEM]: Dr. General Agent is analyzing initial symptoms for {self.state.name}...\n")
            
            # Structured task returning Pydantic schema for routing/extraction
            general_task = Task(
                description=f"""Analyze the patient's context deeply:
                Patient Name: {self.state.name}
                Current Symptoms: {self.state.current_symptoms}
                Medical History: {self.state.medical_history}
                
                STRICT RULES:
                1. If symptoms indicate severe distress, high-risk systemic conditions, or serious potential complications, 
                set 'is_critical' to True and list required labs (like CBC, Chest X-Ray, etc.) in 'suggested_tests'.
                2. If symptoms are mild or seems like minor symptoms like fever, runny nose, sore throat, or routine seasonal diseases, 
                set 'is_critical' to False and provide data in 'prescription_or_diet'.
                3. If 'is_critical' is false which means mild symptoms then :
                i. Write write prescription to the patient in 'doctor_reply' using prescription which you put in 
                'prescription_or_diet'.
                
                4. If 'is_critical' is True which means severe symptoms or symptoms seems dangerous/critical then:
                i. Write your direct response to the patient in 'doctor_reply' saying such as your symptoms seems
                critical so i suggest you some tests ( which you put in 'suggested_test'). You first
                do that and then i will see your report and will give my remarks on.""",
                expected_output="Structured critical assessment and medical guidance.",
                agent=general_agent,
                output_pydantic=DoctorDecisionState
            )

            # Execute Initial Assessment Crew
            crew = Crew(agents=[general_agent], tasks=[general_task], verbose=False)
            result = crew.kickoff().pydantic

            # Dynamic Path Branching
            if result.is_critical:
                self.state.required_tests = result.suggested_tests
                print(f'General Doctor Reply > {result.doctor_reply}')
                print(f'[SYSTEM]: Test ordered: {self.state.required_tests}. Routing to Lab...\n')
                
            else:
                self.state.prescription = result.prescription_or_diet
                print(f'General Doctor Reply > {result.doctor_reply}\n')


    # lab fucntion
    @router(or_(cardiologist_node, orthopedic_node, general_node))
    def lab(self) -> Literal["cardiologist", "orthopedic", "general", "exit_flow"]:

        # Condition A: Agar doctor ne test order kiya hai AUR report abhi tak nahi bani
        if self.state.required_tests and not self.state.report:
            print(f'Performing prescribed tests in Lab : { self.state.required_tests} \n')
            print(f'{self.state.doctor_type} Prescribed the Tests')

            lab_agent = Agent(
                role = "Laboratory Agent ",
                goal = "To simulate the test based on suggested tests by doctor",
                backstory = (
                    """ You are a professional Laboratory Agent specialized in performing suggested tests by doctors.
                     You have 30+ experience of performing tests. """
                ),
                llm = llm,
                verbose = False,
            )

            lab_task = Task(
                description= f"""Your Task is to simulate the suggested tests suggested
                by doctor agent. 
                Here are the suggested test or test of which you have to simulate and generate a realistic test
                report. {self.state.required_tests}.
                
                Instructions to perform task:
                1. Only simulate a test report of the tests suggested by doctors which i mentioned above.
                2. Your test report simulation must look like a real test report.
                3. When you generate/ simulate the tests then stores the simulated report in 'lab_report' and then
                set 'is_report_generated' to True
                4. You do not have to give reply or prescription to user.
                5.Your only task is to simulate tests which are suggested.
                6. In test report, use data like name, age, gender from following states:
                i. use name in report from {self.state.name}.
                ii. Use age in report from {self.state.age}.
                iii. Use gender in report from {self.state.gender}.
                """,
                expected_output = "A text saying tests conducted",
                agent= lab_agent,
                output_pydantic= LabReportState 
            )

            crew = Crew( agents=[lab_agent], tasks=[lab_task], verbose=False)
            result = crew.kickoff().pydantic

            if result.is_report_generated:
                self.state.report = result.lab_report
                print(f'Lab Agent Generated Report : {result.lab_report} suggested by Doctor {self.state.doctor_type.upper()} \n')
                
                # Dynamic Routing: Wapas usi doctor ke paas bhejo jisne test prescribe kiya tha
                if self.state.doctor_type == "cardiologist":
                    return "cardiologist"
                elif self.state.doctor_type == "orthopedic":
                    return "orthopedic"
                else:
                    return "general"
        
        # Condition B: Agar report pehle se maujood hai (yaani doctor report dekh chuka hai) ya case normal tha
        else:
            return "exit_flow"
        
    
    @listen("exit_flow")
    def flow_exit(self):
        print(f'The Flow is Terminated Successfully ')
        
    

flow = ClinicFlow()
flow.kickoff()
flow.plot()