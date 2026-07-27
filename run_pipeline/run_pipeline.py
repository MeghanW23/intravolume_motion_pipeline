import os 
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from MotionCharacterization import CharacterizeIntraVolumeMotion
from MotionCorrection import StartMotionCorrection
from SingleRunfMRIPrep import StartSingleRunfMRIPrep

