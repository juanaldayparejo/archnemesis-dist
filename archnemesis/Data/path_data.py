
ARCHNEMESIS_PATH_PLACEHOLDER='ARCHNEMESIS_PATH/'

def archnemesis_path():
    import os
    nemesis_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'../../')
    return nemesis_path

def archnemesis_resolve_path(path : str):
    if path.startswith(ARCHNEMESIS_PATH_PLACEHOLDER):
        return archnemesis_path() + path[len(ARCHNEMESIS_PATH_PLACEHOLDER):]
    else:
        return path

def archnemesis_indirect_path(path : str):
    if path.startswith(archnemesis_path()):
        return ARCHNEMESIS_PATH_PLACEHOLDER + path[len(archnemesis_path()):]
    else:
        return path