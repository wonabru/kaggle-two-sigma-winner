"""
Internal class which reads SETTINGS.json file and preprocessed it.
"""
import json


class CSettings:
    def __init__(self):
        with open("SETTINGS.json", "r") as settings_file:
            self.settings = json.load(settings_file)
        
        self.copy_to_self(self.settings)
        
    def copy_to_self(self, dic):
        """
        Setting read parameters from SETTING.json file to properties of class
        :param dic: dictionary
        :return:
        """
        for d in dic:
            self.__setattr__(str(d), dic[d])
            
    def print(self):
        """
        saving all settings as string
        :return: string
        """
        return str(self.__dict__)
        
