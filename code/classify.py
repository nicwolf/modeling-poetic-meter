# Administrivia
import data_processing
import re
import nltk
import itertools
import numpy as np
import pickle
import os

# Compile regular expressions
re_patterns = {
	'dipthong' : r'ae|au(?!m)|ei|oe|qui|oi|gui|\biu|iur|iac',
	'stop_liquid' : '[bcdgpt](?=[lr])',
	'two_consonants' : '[^aeiou]{2}',
	'give_elision' : r'h?[aeiou]',
	'take_elision' : r'[aeiou]+m?\b',
	'syllable' : 
		r"""(?ix)
		(.*?(ty|y|ae|au(?!m)|ei|oe|qui|oi|gui|\biu|iur|iac|[aeioy]|(?<!q)u)
		(?:[^aeioutdpbkgq](?=[^aeioyu]|\b)|
		[tdpbkg](?![lraeiou])|
		q(?!u)|\b
		)*)"""
}

# Calculate paths for later use by viterbi algorithm
POSSIBLE_METERS = ['ll','lle','lss','lsse']
ALL_POSSIBLE_PATHS = [
	''.join(p)+'lx' for p in itertools.product(POSSIBLE_METERS,repeat=5)
]

SCANNED_FILE_PATH = '../data/BookOneScansion1_33.txt'
NON_SCANNED_FILE_PATH = '../data/BookOneNoScansion.txt'

def syllable_features(sentence,word_index,syllable_index):
	"""
	Syllables have three kinds of 'traits'. They are ordered here by the
	strength with which they contribute to the determination of the
	length of a syllable. 

	The first array contains information about the possible
	deterministic traits of a syllable (Ultima). 

		0 : First Syllable on Line (Boolean)
			This syllable is long. 

		1 : Second to Last Syllable on Line (Boolean)
			This syllable is long. 

		2 : Last Syllable on Line (Boolean)
			This syllable is indeterminant and can function as either
			a long or a short syllable at the reader's discretion.

	The second array contains information about syllables that end a word. 
	The word ending often lends some insight into the length of the
	syllable (Length by Nature).

		0 : Final -o, -i, -u 
			Usually long.

		1 : Final -as, -es, -us
			Usually long.

		2 : Final -a, -is
			Often short. 

		3 : Final -e
			Usually short.

		4 : Final -us
			Usually short. 

		5 : Final -am, -em, -um
			Always short. 

	The third array contains information about the letters surrounding a 
	vowel and their influence on its length. (Length by Position)

		0 : Followed by Two Consonants
			Usually creates a long vowel.

		1 : Followed by a Stop-Liquid
			Sometimes will make position for a long vowel, not always. 

		2 : Dipthong
			Usually long. 
	"""	
	word = sentence[word_index]
	syllable = word[syllable_index][0]

	# Initialize feature dictionary
	features = {}

	# Determine Ultima traits

	# 1. First syllable in line
	features['first'] = (word_index == 0 and syllable_index == 0)
	# 2. Second to last syllable in line
	features['second_to_last'] = (word_index == (len(sentence) - 1) and syllable_index == (len(word)-2))
	# 3. Last syllable in line
	features['last'] = (word_index == (len(sentence)-1) and syllable_index == (len(word) - 1))

	# Traits useful for finding ellisions

	# 1. If last syllable in word and next word starts with a vowel
	if not word_index == (len(sentence) - 1) and syllable_index == (len(word) - 1):
		next_syllable = sentence[word_index+1][0][0]
		features['can_elide'] = bool(
			re.search(re_patterns['take_elision'], syllable) and
			syllable_index == (len(word) - 1) and 
			re.match(re_patterns['give_elision'], next_syllable)
		)

	# Determine Length by Nature
	if len(syllable) > 1:
		features['final_two'] = syllable[-2:]
	
	features['final_letter'] = syllable[-1]

	# Determine Length by Position
	features['dipthong'] = bool(re.search(re_patterns['dipthong'],syllable))

	if not syllable_index == (len(word)-1):
		next_syllable = word[syllable_index+1][0]
		
		features['followed_stop_liq'] = bool(
			re.match(re_patterns['stop_liquid'],next_syllable)
		)

		features['followed_two_cons'] = bool(
			re.match(re_patterns['two_consonants'],next_syllable)
		)

	else:
		features['followed_stop_liq'] = False
		features['followed_two_cons'] = False

	return features

def line_to_features(line):
	words = line.split()
	temp_line = []
	for word in words:
		syllables = re.findall(re_patterns['syllable'],word)
		# Temp storage for this word
		temp_word = []
		for whole, vowel in syllables:
			temp_word.append((whole.lower(),))		
		temp_line.append(temp_word)
	
	feature_set = []
	for i, word in enumerate(temp_line):
		for j, syl in enumerate(word):
			feature_set.append((syllable_features(temp_line, i, j),))

	return feature_set

def build_classifier():
	# Read in and scan lines from the first book of the Aeneid
	scanned_lines = data_processing.scan_lines(
		SCANNED_FILE_PATH,
		NON_SCANNED_FILE_PATH
	)

	# Build feature sets
	print('{:-^50}'.format('Generating Features'))
	feature_set = []
	for line in scanned_lines:
		for i, word in enumerate(line):
			for j, (syl, length) in enumerate(word):
				feature_set.append((syllable_features(line, i, j), length))

	size = int(len(feature_set)*.25)
	train_set = feature_set[size:]
	test_set = feature_set[:size]
	print("Example features for syllable '{}': ".format(scanned_lines[0][0][0][0]))
	for pair in feature_set[0][0].items():
		print('{:>50}'.format('{} : {}'.format(pair[0],pair[1])))
	print('\n')

	# Train Model
	print('{:-^50}'.format('Classification'))
	print('Training Model...')
	classifier = nltk.NaiveBayesClassifier.train(train_set)

	# Test Model
	print('Testing Model...')
	print('{:>50}'.format(
		'Accuracy: {:.2%}\n'.format(nltk.classify.accuracy(classifier, test_set)))
	)

	# Get most informative features
	classifier.show_most_informative_features(10)
	print('\n')

	print('Generating Confusion Matrix for Naive Classification:')
	# List of what each tag SHOULD be:
	gold = [str(correct) for (features, correct) in test_set]
	# List of what each feature set was tagged as:
	test = [str(classifier.classify(features)) for (features, correct) in test_set]
	cm = nltk.ConfusionMatrix(gold, test)
	print('{:^50}'.format(cm.pretty_format(sort_by_count=True, show_percents=True)))

	print('Using Naive Classification as Prior') 
	print('Probabilities in a Markov Model:')
	# Build an array of true lengths
	lengths = []
	for line in scanned_lines:
		line_lengths = []
		for word in line:
			for (syl, length) in word:
				line_lengths.append(length)
		lengths.append(line_lengths)

	# Get lines from corpus for testing
	num_lines = sum(1 for line in open(SCANNED_FILE_PATH))
	with open(NON_SCANNED_FILE_PATH) as f:
		lines = [f.readline() for n in range(num_lines)]

	# Calculate accuracy of HMM model
	score = 0
	# Build an array of true lengths
	size = int(.25*len(scanned_lines))
	lengths = []
	for line in scanned_lines[:size]:
		line_lengths = []
		for word in line:
			for (syl, length) in word:
				line_lengths.append(length)
		lengths.append(line_lengths)
	total = sum(len(l) for l in lengths)
	for i, line in enumerate(lines[:size]):
		test_labels = classify_line_viterbi(classifier, line)
		for j, label in enumerate(test_labels):
			if label == lengths[i][j]:
				score += 1
		if test_labels == lengths[i]:
			print('yes!')
		print(line)
	
	print('{:>50}'.format('Accuracy: {:.2%}'.format(score/total)))
	return classifier

def classify_line_viterbi(classifier, line):
	# Get features for syllables in line
	features = line_to_features(line)
	# Get number of syllables to figure out possible paths
	num_syllables = len(features)
	possible_paths = [path for path in ALL_POSSIBLE_PATHS if len(path) == num_syllables]
	if possible_paths:
		labels = [classifier.prob_classify(f) for f, in features]
		prob = []
		for path in possible_paths:
			p = 0
			for i, ch in enumerate(path):
				p += np.log(labels[i].prob(ch))
			prob.append((path,p))
		most_likely_path = sorted(prob, key=lambda x: x[1], reverse=True)[0][0]
		return [p for p in most_likely_path]
	else:
		print('Must enter valid dactylic hexameter.')

def label_line(line, classifier):
	labels = classify_line_viterbi(classifier,line)
	words = line.split()
	i = 0
	labelled_line = []
	for word in words:
		labelled_word = ''
		for whole, vowel in re.findall(re_patterns['syllable'], word):
			if labels[i] == 'l':
				labelled_word += re.sub(
					vowel,
					bytes(r'{}\u0304'.format(vowel),'ascii').decode('unicode-escape'),
					whole)
			elif labels[i] == 's':
				labelled_word += re.sub(
					vowel,
					bytes(r'{}\u0306'.format(vowel),'ascii').decode('unicode-escape'),
					whole)
			elif labels[i] == 'e':
				labelled_word += ('(' + whole + ')')
			else:
				labelled_word += whole
			i += 1
		labelled_line.append(labelled_word)
	labelled_line[0] = labelled_line[0][0].upper() + labelled_line[0][1:]
	
	return ' '.join(labelled_line)

def label_line_latex(line, classifier):
	labels = classify_line_viterbi(classifier,line)
	words = line.split()
	i = 0
	labelled_line = []
	for word in words:
		labelled_word = ''
		for whole, vowel in re.findall(re_patterns['syllable'], word):
			if labels[i] == 'l':
				labelled_word += re.sub(
					vowel,
					(r'\={' + bytes(r'{}'.format(vowel),'ascii').decode('unicode-escape') + '}'),
					whole)
			elif labels[i] == 's':
				labelled_word += re.sub(
					vowel,
					r'\u{' + bytes(r'{}'.format(vowel),'ascii').decode('unicode-escape') + '}',
					whole)
			elif labels[i] == 'e':
				labelled_word += ('(' + whole + ')')
			else:
				labelled_word += whole
			i += 1
		labelled_line.append(labelled_word)
	labelled_line[0] = labelled_line[0][0].upper() + labelled_line[0][1:]
	
	return ' '.join(labelled_line)

def main():
	if os.path.exists('aeneid_classifier'):
		with open('aeneid_classifier','rb') as f:
			classifier = pickle.load(f)
	else:
		classifier = build_classifier()
		with open('aeneid_classifier','wb') as f:
			pickle.dump(classifier,f)
	
	# while True:
	# 	input_line = input("Enter line ('q' to quit): ")
	# 	if input_line == 'q':
	# 		break
	# 	output_line = label_line(input_line, classifier)
	# 	print('Scanned: {}\n'.format(output_line))

	with open(NON_SCANNED_FILE_PATH, 'r') as f:
		raw = f.readlines()

	with open('test_output_naive.txt', 'w') as f:
		for i in range(10):
			line = raw[i]
			features = line_to_features(line)
			labels = [classifier.classify(f) for f, in features]
			words = line.split()
			i = 0
			labelled_line = []
			for word in words:
				labelled_word = ''
				for whole, vowel in re.findall(re_patterns['syllable'], word):
					if labels[i] == 'l':
						labelled_word += re.sub(
							vowel,
							(r'\={' + bytes(r'{}'.format(vowel),'ascii').decode('unicode-escape') + '}'),
							whole)
					elif labels[i] == 's':
						labelled_word += re.sub(
							vowel,
							r'\u{' + bytes(r'{}'.format(vowel),'ascii').decode('unicode-escape') + '}',
							whole)
					elif labels[i] == 'e':
						labelled_word += ('(' + whole + ')')
					else:
						labelled_word += whole
					i += 1
				labelled_line.append(labelled_word)
			labelled_line[0] = labelled_line[0][0].upper() + labelled_line[0][1:]
			f.write(' '.join(labelled_line) + '\\\\ \n')

	with open('test_output_viterbi.txt', 'w') as f:
		for i in range(10):
			line = raw[i]
			f.write(label_line_latex(line,classifier)+'\\\\ \n')
			

if __name__ == '__main__':
	main()