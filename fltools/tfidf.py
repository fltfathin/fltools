
import numpy as np
import logging as log
log.basicConfig(format='%(asctime)s [%(levelname)7s] %(message)s', level=log.DEBUG)



def test():
	pass

class tf_idf:
	input_tokens = []
	#
	unique_words = []
	#
	tf_vector = []
	idf_vector = []
	weight_vector = []
	#
	
	def __init__(self, tf_method = None, idf_method = None):
		"""
		tf-idf for similarity calculation purpose
		"""
		if tf_method is None:
			self.tf_method = self.tf_raw
		else:
			self.tf_method = tf_method
		if idf_method is None:
			self.tf_method = self.idf_norm
		else:
			self.idf_method = idf_method

	def pretty_solve(self, documents):
		log.debug(f"document: {documents}")
		self.input_tokens = documents
		self.get_unique_words()
		log.debug(f"unique_words: {self.unique_words}")
		self.calculate_tfidf()
		log.debug(f"tfidf: {self.tfidf_vector}")
		self.calculate_cosine()

	def solve(self, documents):
		self.input_tokens = documents
		self.get_unique_words()
		self.calculate_tfidf()
		return self.calculate_cosine()


	@staticmethod
	def tf_raw(term, document):
		tf = 0
		for word in document:
			if word == term:
				tf += 1
		return tf

	@staticmethod
	def idf_norm(term,documents):
		N = len(documents)
		df = 0
		for document in documents:
			for term in document:
				df += 1
		return np.log10(N/np.abs(df))

	def calculate_cosine(self):
		size = len(self.input_tokens)
		cosine_vector = np.zeros((size,size))
		for i in range(size):
			for j in range(size):
				if i == j:
					cosine_value = 1
				elif i > j:
					cosine_value = cosine_vector.item((j,i))
				else:
					cosine_value = self.cosine_pair((0,1))
				cosine_vector.itemset((i,j),cosine_value)
		return cosine_vector
			

	def cosine_pair(self,pair):
		s1,s2 = pair
		num = 0
		denum1 = 1
		denum2 = 1
		for i in range(len(self.unique_words)):
			num += self.tfidf_vector.item((s1,i)) * self.tfidf_vector.item((s2,i))
			denum1 += self.tfidf_vector.item((s1,i)) ** 2
			denum2 += self.tfidf_vector.item((s2,i)) ** 2
		similarity  = num / (np.sqrt(denum1)*np.sqrt(denum2))
		return similarity

	def calculate_tfidf(self):
		self.tf_vector = np.zeros((len(self.input_tokens),len(self.unique_words)))
		self.idf_vector = np.zeros((1,len(self.unique_words)))
		self.df_vector = np.zeros((1,len(self.unique_words)))
		self.tfidf_vector = np.zeros((len(self.input_tokens),len(self.unique_words)))
		for d, document in enumerate(self.input_tokens):
			for t, term in enumerate(self.unique_words):
				if d == 0:
					df, idf = self.idf_method(term, self.input_tokens)
					self.df_vector.itemset((0,t),df)
					self.idf_vector.itemset((0,t),idf)
				else:
					df = self.df_vector.item((0,t))
					idf = self.idf_vector.item((0,t))
				tf = self.tf_method(term, document)
				self.tf_vector.itemset((d,t),tf)

				tfidf = tf * idf
				self.tfidf_vector.itemset((d,t),tfidf)

		log.debug(self.tf_vector)


	def get_unique_words(self):
		self.unique_words = []
		for document in self.input_tokens:
			for word in document:
				if not word in self.unique_words:
					self.unique_words.append(word)
		return self.unique_words

def main():
	tcalc = tf_idf()
	tcalc.pretty_solve([["rawr","hewwo","this","hello","world","hello"],["hello","world","this","is","me"]])


if __name__ == '__main__':
	main()
