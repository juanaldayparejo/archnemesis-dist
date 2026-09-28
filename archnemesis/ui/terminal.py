import sys

ui_show = lambda x: print(x)

def str_yes_or_no(response : str, default : bool | None = None) -> bool | None:
    if len(response) == 0:
        return default
    if response[0] in ('y','Y'):
        return True
    if response[0] in ('n', 'N'):
        return False
    return None


def ui_ask_yn(msg : str, default : bool | None = None, msg_yes : str = '', msg_no : str = '', msg_interrupt : str = 'User cancelled interaction, EXITING.') -> bool:
    yn = None
    
    while yn is None:
        prompt = f'{msg} ({"Y" if default is not None and default else "y"}/{"N" if default is not None and not default else "n"}) > '
        try:
            response = input(prompt).strip()
        except KeyboardInterrupt:
            ui_show(f'\n  {msg_interrupt}')
            sys.exit()
        yn = str_yes_or_no(response, default=default)
        
        if yn is None:
            ui_show(f'  Unknown response "{response}".')
            ui_show( '  Answer "y" or "n", or press [return] for default choice denoted by capital letter.')
    
    if yn:
        ui_show(f'  {msg_yes}')
    else:
        ui_show(f'  {msg_no}')
    
    return yn