import enum

class Datasets(enum.Enum):
    multi_center = "multi_center"
    dl4ivf = "dl4ivf"
    
class Tasks(enum.Enum):
    TE = "TE"
    ICM = "ICM"
    EXP = "EXP"

class EXP_Annot(enum.Enum):
    one = 1
    two = 2
    three = 3
    four = 4
    five = 5
    six = 6

class ICM_Annot(enum.Enum):
    A = "A"
    B = "B"
    C = "C"

class TE_Annot(enum.Enum):
    A = "A"
    B = "B"
    C = "C"

annot_enum_dic = {Tasks.EXP:EXP_Annot,Tasks.ICM:ICM_Annot,Tasks.TE:TE_Annot}