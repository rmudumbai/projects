import os
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
#from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import PCA as PCAmllib

# Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# Load documents (one per line).
# Read file with Python to avoid Java security issues with newer Java versions
script_dir = os.path.dirname(os.path.abspath(__file__))
tsv_path = os.path.join(script_dir, "subset-small.tsv")
with open(tsv_path, "r", encoding="utf-8") as f:
    lines = [line.rstrip('\n\r') for line in f.readlines()]
rawData = sc.parallelize(lines)
fields = rawData.map(lambda x: x.split("\t"))
documents = fields.map(lambda x: x[3].split(" "))

# Store the document names for later:
documentNames = fields.map(lambda x: x[1])

# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# At this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value.

# Let's compute the TF*IDF of each term in each document:
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.
model = PCAmllib(2).fit(tfidf)
pc = model.transform(tfidf)

#mat = RowMatrix(tfidf)
# Calculate PCA
#pc = mat.computePrincipalComponents(int(mat.numCols))

print("Principal components :")
print(pc)
