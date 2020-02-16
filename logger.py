"""
logging class
"""
import logging
import os


class CLogger:
    def __init__(self, path_to_logs):
        _path_to_log = os.path.realpath(path_to_logs)
        self.create_folder(_path_to_log)
        _path_to_log = os.path.join(_path_to_log, "logs.txt")
        logging.basicConfig(filename=_path_to_log,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        
    def create_folder(self, folder_name):
        """
        creating directory for given name
        :param folder_name: string
        :return:
        """
        _path = os.path.realpath(folder_name)
        if os.path.isdir(_path) == False:
            try:
                os.mkdir(_path)
            except Exception as ex:
                print("Could not create directory: {}".format(_path))
                raise ex
            
    def info(self, txt):
        """
        informative logging
        :param self: CLog class object
        :param txt: string
        :return:
        """
        print(txt)
        logging.info(txt)
        
    def error(self, txt):
        """
        error logging with raising exception
        :param self: CLog class object
        :param txt: string
        :return:
        """
        print(txt)
        logging.error(txt)
        raise Exception(txt)