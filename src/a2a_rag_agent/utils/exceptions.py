from fastapi import HTTPException


class AppException(HTTPException):

    def __init__(self, status_code, detail=None, headers=None):
        super().__init__(status_code, detail, headers)
