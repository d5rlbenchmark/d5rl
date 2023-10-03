from .pick_place import PickPlace
from .drawer_open import DrawerOpen
from .grasp import Grasp
from .button_press import ButtonPress
from .pick_place_miss import PickPlaceMiss
from .bin_sort import BinSort
from .bin_sort_neutral import BinSortNeutral
from .bin_sort_neutral_stored import BinSortNeutralStored
from .stitching import Stitching
from .bin_sort_mult import BinSortNeutralMult, BinSortMult
from .bin_sort_mult_stored import BinSortNeutralMultStored, BinSortMultStored
from .bin_sort_mult_storedseg import BinSortNeutralMultStoredSeg, BinSortMultStoredSeg

policies = dict(
    grasp=Grasp,
    pickplace=PickPlace,
    drawer_open=DrawerOpen,
    button_press=ButtonPress,
    pickplacemiss=PickPlaceMiss,
    binsort=BinSort,
    binsortneutral=BinSortNeutral,
    binsortneutralstored=BinSortNeutralStored,
    stitching=Stitching,
    binsortmult=BinSortMult,
    binsortneutralmult=BinSortNeutralMult,
    binsortneutralmultstored=BinSortNeutralMultStored,
    binsortmultstored=BinSortMultStored,
    binsortmultstoredseg=BinSortMultStoredSeg,
    binsortneutralmultstoredseg=BinSortNeutralMultStoredSeg,
)
