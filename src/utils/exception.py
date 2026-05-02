import sys

class CustomException(Exception):
    def __init__(self, error, error_detail: sys):
        _, _, tb = error_detail.exc_info()
        super().__init__(f"{error} at line {tb.tb_lineno}")