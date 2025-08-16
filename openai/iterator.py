from typing import Iterable, Any
from dataclasses import dataclass


@dataclass
class State:
    index: int


class ResumableIterator:
    def __init__(self, iterable: Iterable[Any]) -> None:
        self._data: list[Any] = list(iterable)  # Store items as a list for indexing
        self._index: int = 0  # Current position

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self._index >= len(self._data):
            raise StopIteration
        item = self._data[self._index]
        self._index += 1
        return item

    def get_state(self) -> State:
        # Return a State object representing the current index
        return State(self._index)

    def set_state(self, state: State) -> None:
        # Restore the iterator to the given index using a State object
        if not isinstance(state, State):
            raise TypeError("state must be a State instance")
        if not (0 <= state.index <= len(self._data)):
            raise ValueError("Invalid state index")
        self._index = state.index


# Regular test code
def test() -> None:
    items = ["a", "b", "c", "d"]
    it = ResumableIterator(items)
    print(next(it))  # 'a'
    print(next(it))  # 'b'
    state = it.get_state()
    print(next(it))  # 'c'
    it.set_state(state)  # Resume from after 'b'
    print(next(it))  # 'c'
    print(next(it))  # 'd'
    try:
        print(next(it))  # Should raise StopIteration
    except StopIteration:
        print("StopIteration raised as expected")

    # Test invalid state
    try:
        it.set_state(State(100))
    except ValueError as e:
        print("Caught ValueError as expected:", e)

    # Test type check
    try:
        it.set_state(5)
    except TypeError as e:
        print("Caught TypeError as expected:", e)


test()
