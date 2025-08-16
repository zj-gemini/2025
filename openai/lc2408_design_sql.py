from typing import List
from collections import defaultdict
import dataclasses


@dataclasses.dataclass
class Table:
    n_col: int
    rows: defaultdict[int, List[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(List[str])
    )
    next_id: int = 1


class SQL:

    def __init__(self, names: List[str], columns: List[int]):
        self.tables = defaultdict(Table)
        if len(names) != len(columns):
            raise ValueError("blabla")
        for name, n_col in zip(names, columns):
            self.tables[name] = Table(n_col)

    def ins(self, name: str, row: List[str]) -> bool:
        if name not in self.tables or len(row) != self.tables[name].n_col:
            return False
        t = self.tables[name]
        t.rows[t.next_id] = row
        t.next_id += 1
        return True

    def rmv(self, name: str, rowId: int) -> None:
        if name in self.tables and rowId in self.tables[name].rows:
            del self.tables[name].rows[rowId]

    def sel(self, name: str, rowId: int, columnId: int) -> str:
        invalid_str = "<null>"
        if name not in self.tables:
            return invalid_str
        t = self.tables[name]
        if rowId not in t.rows:
            return invalid_str
        r = t.rows[rowId]
        if columnId >= 1 and columnId <= len(r):
            return r[columnId - 1]
        return invalid_str

    def exp(self, name: str) -> List[str]:
        if name not in self.tables:
            return []
        t = self.tables[name]
        rst = []
        for row_id in sorted(t.rows):
            r = t.rows[row_id]
            rst.append(",".join([str(row_id)] + r))
        return rst


def test():
    sql = SQL(["one", "two", "three"], [2, 3, 1])
    print(sql.ins("two", ["first", "second", "third"]))  # True
    print(sql.tables)
    print(sql.sel("two", 1, 3))  # "third"
    return
    print(sql.ins("two", ["fourth", "fifth", "sixth"]))  # True
    print(sql.exp("two"))  # ["1,first,second,third", "2,fourth,fifth,sixth"]
    sql.rmv("two", 1)
    print(sql.sel("two", 2, 2))  # "fifth"
    print(sql.exp("two"))  # ["2,fourth,fifth,sixth"]
    print(sql.ins("two", ["fourth", "fifth"]))  # False
    print(sql.sel("two", 1, 2))  # "<null>"
    print(sql.exp("four"))  # []


# Uncomment to run tests
test()
