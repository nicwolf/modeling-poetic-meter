syllable_pattern = re.compile(
	r"""(?ix)
	(.*?(ty|y|ae|au(?!m)|ei|oe|qui|oi|gui|\biu|iur|iac|[aeioy]|(?<!q)u)
	(?:[^aeioutdpbkgq](?=[^aeioyu]|\b)|
	[tdpbkg](?![lraeiou])|
	q(?!u)|\b
	)*)
	"""
)
