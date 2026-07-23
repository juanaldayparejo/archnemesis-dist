

ui_show = lambda x: print(x)

def str_yes_or_no(response : str, default : bool | None = None) -> bool | None:
    response = response.strip()
    if len(response) == 0:
        return default
    if response[0] in ('y','Y'):
        return True
    if response[0] in ('n', 'N'):
        return False
    return None


def ui_ask_yn(msg : str, default : bool | None = None) -> bool:
    yn = None
    
    while yn is None:
        prompt = f'{msg} ({"Y" if default is not None and default else "y"}/{"N" if default is not None and not default else "n"}) >'
        response = input(prompt).split()
        yn = str_yes_or_no(response, default=True)
        
        if yn is None:
            ui_show(f'  Unknown response "{response}".')
            ui_show( '  Answer "y" or "n", or press [return] for default choice denoted by capital letter.')
    
    return yn