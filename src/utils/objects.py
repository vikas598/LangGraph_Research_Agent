from pydantic import BaseModel, Field
from typing import List

#creating analyst object
class Analyst(BaseModel):
    affiliation:str = Field(description="Primary affiliation of the Analyst")
    name:str = Field(description="Name of the analyst.")
    role:str = Field(description="Role of the analyst in context of the topic")
    description:str = Field(description="Description of the analyst focus, concern and motives")

    @property
    def persona(self)->str:
        return f"Name : {self.name}\n Role: {self.role} \n Affiliation: {self.affiliation} \n Description: {self.description}"

class Perspectives(BaseModel):
    analysts : List[Analyst] = Field(description=" Comprehensive list of analysts with their role, name, affiliation and description")