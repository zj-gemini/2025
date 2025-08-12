# Examples of dataclasses with default constructors

from dataclasses import dataclass, field
from typing import List


# Dataclass with mutable default (use field)
@dataclass
class Group:
    members: List[str] = field(default_factory=list)


g1 = Group()
g2 = Group(["Alice", "Bob"])
g1.members.append("Carol")
print(g1)  # Group(members=['Carol'])
print(g2)  # Group(members=['Alice', 'Bob'])

# Examples for abc (Abstract Base Classes)
from abc import ABC, abstractmethod


class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass


class Dog(Animal):
    def speak(self):
        return "Woof!"


class Cat(Animal):
    def speak(self):
        return "Meow!"


dog = Dog()
cat = Cat()
print(dog.speak())  # Woof!
print(cat.speak())  # Meow!

# Trying to instantiate Animal directly will raise an error:
# animal = Animal()  # TypeError: Can't instantiate abstract class Animal with
