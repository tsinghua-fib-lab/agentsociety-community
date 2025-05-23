from agentsociety.agent import CitizenAgentBase

__all__ = ["ExampleCitizen"]

class ExampleCitizen(CitizenAgentBase):
    def __init__(self, name: str, age: int, gender: str):
        # TODO: build a real example citizen
        super().__init__(name, age, gender)
        self.name = name
        self.age = age
        self.gender = gender
