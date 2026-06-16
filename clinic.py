from crewai import Agent, Task, Crew
from crewai.flow.flow import Flow, start, router, listen
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




class ClinicFlow(Flow[MyClinicStates]):

    @start()

    def greet_and_validation(self):
        print("============================================================")
        print("🏥 WELCOME TO THE AI CLINIC INFORMATION DESK 🏥")
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
                    
                    print(f"🎉 [SYSTEM]: New Patient successfully saved to CSV Database!")
                    print(f"⚠️ [SYSTEM]: Generated Patient ID is: {self.state.patient_id}\n")

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
    def cardiologist_node(self):
        print("\n🏥 --- WELCOME TO THE CARDIOLOGY DEPARTMENT --- 🏥")
        print(f"Dr. Heart Agent: Hello {self.state.name}, I am your Cardiologist.")
        print(f"Reviewing Symptoms: '{self.state.current_symptoms}'")
        print("Action: Running an immediate ECG check. Please wait...")
        return "Cardiologist treatment started."

    @listen("orthopedic")
    def orthopedic_node(self):
        print("\n🏥 --- WELCOME TO THE ORTHOPEDIC DEPARTMENT --- 🏥")
        print(f"Dr. Bones Agent: Hello {self.state.name}, I am your Orthopedic Specialist.")
        print(f"Reviewing Symptoms: '{self.state.current_symptoms}'")
        print("Action: Scheduling an immediate X-Ray. Please relax...")
        return "Orthopedic treatment started."

    @listen("general")
    def general_node(self):
        print("\n🏥 --- WELCOME TO THE GENERAL PHYSICIAN DEPARTMENT --- 🏥")
        print(f"Dr. General Agent: Hello {self.state.name}, I am your General Physician.")
        print(f"Reviewing Symptoms: '{self.state.current_symptoms}'")
        print("Action: Checking vitals and writing a general prescription...")
        return "General treatment started."
    


flow = ClinicFlow()

flow.kickoff()
flow.plot()
