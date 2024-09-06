XTB_BIN = 'D:/programs/xtb-6.6.1/bin/xtb.exe'
XTB_PARAM_DIR = 'D:/programs/xtb-6.6.1/share/xtb'
G16_ROOT = 'C:/G16W'
G16_SCRATCH_DIR = 'D:/scratch'
VIEWER_PATH = 'D:/programs/jmol/jmol.bat'
TEXT_EDITOR_PATH = 'C:/Windows/notepad.exe'

ASE_OMP_NUM_THREADS = None  # integer or None (auto)
ASE_OMP_STACKSIZE = '256M'  # memory per thread (without B)

G16_TEMPLATE_FILE_NAME = 'template.gjf'

CHECK_INTERVAL = 2000  # in msec
DEFAULT_NOTIFY_FINISHED = True

# control parameter list
INTERPOLATION_METHOD_LIST = ['idpp', 'linear']
NEB_METHOD_LIST = ['aseneb', 'improvedtangent', 'eb', 'spline', 'string']
NEB_OPTIMIZER_LIST = ['fire', 'lbfgs', 'lbfgslinesearch', 'composite']
XTB_GFN_LIST = ['gfn0', 'gfn1', 'gfn2', 'gfnff']
XTB_SOLVENT_LIST = [
    'None',
    'Acetone',
    'Acetonitrile',
    'Aniline',
    'Benzaldehyde',
    'Benzene',
    'CH2Cl2',
    'CHCl3',
    'CS2',
    'Dioxane',
    'DMF',
    'DMSO',
    'Ether',
    'Ethylacetate',
    'Furane',
    'Hexadecane',
    'Hexane',
    'Methanol',
    'Nitromethane',
    'Octanol',
    'Phenol',
    'Toluene',
    'THF',
    'Water'
]

# default settings for GUI
DEFAULT_PROJECT_NAME = 'aseneb'
DEFAULT_OPT_INIT = True
DEFAULT_OPT_FINAL = True
DEFAULT_NUM_IMAGES = 10
DEFAULT_INTERPOLATION_METHOD = 'idpp'
DEFAULT_NEB_METHOD = 'aseneb'
DEFAULT_NEB_K = 0.1
DEFAULT_NEB_CLIMB = True
DEFAULT_NEB_OPTIMIZER = 'fire'
DEFAULT_NEB_FMAX = 0.05
DEFAULT_NEB_STEPS = 1000
DEFAULT_NEB_PARALLEL = 1
DEFAULT_CALCULATOR_TYPE = 'xtb'
DEFAULT_XTB_GFN = 'gfn2'
DEFAULT_XTB_CHARGE = 0
DEFAULT_XTB_UHF = 0
DEFAULT_XTB_SOLVATION = None
DEFAULT_XTB_SOLVENT = None
DEFAULT_XTB_CPU = 2
DEFAULT_XTB_MEMORY = '500M'
DEFAULT_G16_CPU = 1
DEFAULT_G16_MEMORY = '1GB'
DEFAULT_G16_GUESS_ADDITIONAL_KEYWORDS = 'stable=opt guess=mix'


# pre-optimization conditions
PREOPT_FMAX = 0.005
PREOPT_STEPS = 1000
