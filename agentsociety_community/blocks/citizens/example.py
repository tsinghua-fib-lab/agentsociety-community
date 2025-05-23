from agentsociety.agent import Block

__all__ = ["ExampleCitizenBlock"]

class ExampleCitizenBlock(Block):
    def __init__(self, name: str, age: int, gender: str):
        # TODO: build a real example citizen
        super().__init__(name, age, gender)
        self.name = name
        self.age = age
        self.gender = gender
