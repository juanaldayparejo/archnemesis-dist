


from bs4 import Tag, NavigableString

from archnemesis.Data.isotope_data import standard_isotope_aliases


def parse_html_float(items):
	print(f'{items=}')
	if items[0].endswith('\xa0×\xa010'):
		x = float(items[0][:len('\xa0×\xa010')])
	else:
		x = float(items[0])
	if len(items) == 2 and items[1].name == 'sup':
		return x * 10**(int(items[1].text))
	else:
		return x


def html_to_molecular_formula(html_content_or_tag : Tag | list[NavigableString|Tag]) -> str:
	print(f'{html_content_or_tag=}')
	# Turn all of the content into a standard form, then read as string
	if isinstance(html_content_or_tag, Tag):
		html_content = html_content_or_tag.content
	else:
		html_content = html_content_or_tag
	
	formula = []
	within_mol_def = False
	
	for item in html_content:
		print(f'DEBUG : {type(item)=} {item=}')
		if isinstance(item, Tag):
			if item.name == 'sup':
				if item.text.endswith('+') or item.text.endswith('-'):
					if within_mol_def:
						within_mol_def = False
					
					if not within_mol_def:
						txt = str(item.text)
						formula.append(f'{txt[-1]}{txt[:-1]}')
						
				else:
					raise RuntimeError('Must not encounter non-charge superscript')
					
			if item.name == 'sub':
				if not within_mol_def:
					print(f'DEBUG : {formula=}')
					raise RuntimeError('Must be within mol def when encountering subscript')
				else:
					formula.append(f'{item.text}')
					within_mol_def = False
			
		if isinstance(item, (str,NavigableString)):
			names = []
			for char in item:
				if char.isupper():
					names.append(char)
				else:
					names[-1] += char
			for name in names:
				formula.append(f'{name}')
				within_mol_def = True
	
	if within_mol_def:
		formula.append('')
	return ''.join(formula)



def html_to_isotopologue_formula(html_content_or_tag : Tag | list[NavigableString|Tag]) -> str:
	# Turn all of the content into a standard form, then read as string
	if isinstance(html_content_or_tag, Tag):
		html_content = html_content_or_tag.content
	else:
		html_content = html_content_or_tag
	
	formula = []
	within_iso_def = False
	#expect_list = ('nucleon_number', 'atom_name', 'atom_count', 'charge')
	
	expect = 'nucleon_number'
	for item in html_content:
		print(f'DEBUG : {type(item)=} {item=}')
		if isinstance(item, Tag):
			if item.name == 'sup':
				if item.text.endswith('+') or item.text.endswith('-'):
					if within_iso_def:
						formula.append(')')
						within_iso_def=False
					
					if not within_iso_def:
						txt = str(item.text)
						formula.append(f'{txt[-1]}{txt[:-1]}')
				else:
					if within_iso_def:
						formula.append(')')
						within_iso_def = False
					if not within_iso_def:
						formula.append(f'({item.text}')
						within_iso_def = True
						expect = 'atom_name'
				
					
			if item.name == 'sub':
				if within_iso_def:
					formula.append(f'){item.text}')
					within_iso_def = False
					expect = 'nucleon_number'
				else:
					raise RuntimeError('Must be within iso def when encountering subscript')
			
		if isinstance(item, (str,NavigableString)):
			names = []
			for char in item:
				if char.isupper():
					names.append(char)
				else:
					names[-1] += char
			for name in names:
				print(f'DEBUG : {within_iso_def=} {expect=} {name}')
				if expect != 'atom_name' and within_iso_def:
					formula.append(f')({standard_isotope_aliases[name]}')
					within_iso_def = True
					expect = 'atom_count'
				elif not within_iso_def:
					formula.append(f'({standard_isotope_aliases[name]}')
					within_iso_def = True
					expect = 'atom_count'
				else:
					formula.append(f'{name}')
					within_iso_def = True
					expect = 'atom_count'
	
	if within_iso_def:
		formula.append(')')
	return ''.join(formula)















