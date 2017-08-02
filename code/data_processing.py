# Administrivia
import unicodedata
import re

SCANNED_FILE_PATH = '../data/BookOneScansion1_33.txt'
NON_SCANNED_FILE_PATH = '../data/BookOneNoScansion.txt'

syllable = re.compile(
	r"""(?ix)
	(.*?(ty|y|ae|au(?!m)|ei|oe|qui|oi|gui|\biu|iur|iac|[aeioy]|(?<!q)u)
	(?:[^aeioutdpbkgq](?=[^aeioyu]|\b)|
	[tdpbkg](?![lraeiou])|
	q(?!u)|\b
	)*)""")

def countDiacritics(file):
	with open(file,'r') as f:
		text = f.readlines()
	lengths = []
	for line in text:
		for ch in line:
			if '0304' in unicodedata.decomposition(ch):
				lengths.append('l')
			elif '0306' in unicodedata.decomposition(ch):
				lengths.append('s')
			elif ch == '(':
				lengths.append('e')
		lengths.append('x')
	return lengths

def scan_lines(scanned_file_path, non_scanned_file_path):
	"""
	This function returns an array of labelled data. 
	"""

	# First, extract the number of lines and correct metrical scansion patterns from
	# the scanned corpus. We could process the same corpus, removing all metrical
	# information, but it will be easier to just grab the correct number of lines
	# from the non-scanned corpus. 
	
	print('{:-^50}'.format('Generating Corpus'))
	print('Loading correct metrical scansion from file...')

	# Get lengths from scanned file
	lengths = countDiacritics(SCANNED_FILE_PATH)

	# Get number of scanned lines
	num_lines = sum(1 for line in open(SCANNED_FILE_PATH))
	print('Total number of lines: {}'.format(num_lines))

	# Read lines from non-scanned file
	print('Opening non-scanned text...')
	with open(NON_SCANNED_FILE_PATH,'r') as f:
		lines = [f.readline() for n in range(num_lines)]

	# Zip together syllables the lengths extracted from the scanned text, we want
	# to preserve newlines and words so our target array is a ragged 3d array.
	# The whole poem is an array of lines, each line has an array of words, and
	# each word has an array of syllables. 
	# 
	# [
	# 	[
	# 		[('Ar',0), ('ma',1)],...,('is',X)], ... 
	# 	]
	# ]
	# 
	# This is acheived by a nested iteration through lines and syllables and
	# a constant progression through the indices of lengths

	# Start counter for indexing into lengths
	i = 0 

	# Initialize scanned line array - this is the correct data to build
	# train/test sets from
	scanned_lines = []

	print('Syllabifying and labelling...')
	for line in lines:
		line = re.sub(r':|\.|;|\?|,', '', line)
		words = line.split()
		temp_line = []
		for word in words:
			syllables = re.findall(syllable,word)
			# Temp storage for this word
			temp_word = []
			for whole, vowel in syllables:
				temp_word.append((whole.lower(), lengths[i]))
				i += 1			
			temp_line.append(temp_word)
		scanned_lines.append(temp_line)
	num_syllables = sum(
		1 for line in scanned_lines for word in line for syllable in word
	)
	print('Number of syllables: {}'.format(num_syllables))


	print('First line: {}\n'.format(scanned_lines[0]))

	return scanned_lines







