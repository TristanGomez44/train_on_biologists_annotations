import enum

class Datasets(enum.Enum):
    multi_center = "multi_center"
    dl4ivf = "dl4ivf"
    
class Tasks(enum.Enum):
    TE = "TE"
    ICM = "ICM"
    EXP = "EXP"

class EXP_Annot(enum.Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    NULL = "Null"


class ICM_Annot(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    NULL = "Null"

class TE_Annot(enum.Enum):
    A = "A"
    B = "B"
    C = "C"
    NULL = "Null"

annot_enum_dic = {Tasks.EXP:EXP_Annot,Tasks.ICM:ICM_Annot,Tasks.TE:TE_Annot}