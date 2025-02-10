import json


class Config:
    url: str = None

    def __init__(self, test: bool = True):
        self.test = test
        with open("config.json") as f:
            self.data = json.load(f)
        if test == True:
            self.url = self.data["api-info"]["cert"]
            self.account_number = self.data["account-numbers"]["cert"]
        else:
            self.url = self.data["api-info"]["prod"]
            self.account_number = self.data["account-numbers"]["prod"]

        self.username = self.data["personal-data"]["login"]
        self.password = self.data["personal-data"]["password"]