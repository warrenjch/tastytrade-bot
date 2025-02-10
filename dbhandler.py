import sqlite3
import pandas as pd
from typing import Literal

class DBHandler:
    def __init__(self):
        self.conn = sqlite3.connect('./dump/tastytrade.db')
        self.cursor = self.conn.cursor()

    def modify_table(self, table_name: str, jsonobject: dict, if_exists:Literal["fail","replace","append"]='fail', method:Literal["multi"]='multi') -> None:
        df = pd.DataFrame(jsonobject)
        df.to_sql(name=table_name, con=self.conn, if_exists=if_exists, method=method)