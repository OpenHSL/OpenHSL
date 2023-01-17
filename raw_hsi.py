
class Raw_Data:
    """
    Итератор сырых данных с гипера
    При вызове next возвращает lambda-y слои на каждой итерации
    """
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration
