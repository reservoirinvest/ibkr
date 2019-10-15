# jup2py.py
# program to convert functions in cells to indivdiual .py files
# removes magic commands and asyncio conflicting code

# NOTE: the first line of Module's cell to be converted should have the following:
# <function_name>.py

"""Program to convert .ipynb to .py files
Date: 07-Oct-2019
Ver: 1.0
"""

import json
from os import listdir

fs = listdir() # get the filenames

# make an exclusion list
magic_words = [s for s in str(magics).split(' ') if s not in '' if '%' in s[0]]
magic_words = [s for s in magic_words if s != '%']


# searchfor = ['helper.ipynb', 'nse_func.ipynb', 'snp_func.ipynb', 'nse_main.ipynb', 'snp_main.ipynb']  # list of files to be converted into
# ipfilelist = [f for f in fs if any(word in f for word in searchfor)]

# remove unwanted file extensions
ipfilelist = [f for f in fs if f[-5:] == 'ipynb' if f[:1] not in ['_'] if f[:-6] not in ['___test']]

for file in ipfilelist:
    code_cells = []  #initialize code_cells
    with open(file) as datafile:
        code = json.load(datafile)
        code_cells.append([cell['source'] for cell in code['cells'] if cell['cell_type'] == 'code'])
        codes = [cell for cells in code_cells for cell in cells if cell]
        code_dict = {pycode[0][2:-1]:pycode for pycode in codes if pycode[0][-4:] == '.py\n'}
        with open(file[:-6]+'.py', 'w') as f:
            for k, v in code_dict.items():
                for line in v:
                    if not any(word in line for word in exclude):
                        f.write(line)        
                f.write('\n\n#_____________________________________\n\n')
        

#_____________________________________

