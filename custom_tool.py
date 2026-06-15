from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Type
import os
import csv


# defining input schema for my tool
class Mytoolinput(BaseModel):
    path : str = Field(..., description= " the path to which data will be read from and write to")

# defining my custom tool
class CSVReadWriteTool(BaseTool):
    name : str  = "Csv read and write tool"
    description : str = " this tool will read csv file and write data to it"
    args_schema : Type[BaseModel] = Mytoolinput

    def _run(self, path : str)->str:

        if not os.path.exists(path):

            try:
                # create a blank csv file 
                with open(path, mode='w', encoding="utf-8", newline="") as file:
                    pass
                print(f' File created successfully on path {path}')
                return None
            
            except Exception as e:
                print(f"Error creating file: {e}")
                return None
            
        else:

            try :
               with open(path, mode="r", encoding="utf-8-sig") as file:
                    reader = csv.DictReader(file)
                    data = list(reader)
                    
               print(f"Success: File found and read successfully. Total rows: {len(data)}")
               return data
            
            except Exception as e:
                print(f"Error reading file: {e}")
                return None
        
