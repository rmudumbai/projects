import os
import re
import warnings
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

# Boilerplate Spark stuff:
# Suppress Python warnings
warnings.filterwarnings('ignore')
# Suppress hostname resolution warning by setting SPARK_LOCAL_IP
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_SUBMIT_ARGS'] = '--driver-java-options="-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR" pyspark-shell'
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF") \
    .set("spark.driver.host", "localhost") \
    .set("spark.sql.warehouse.dir", "file:///tmp/spark-warehouse") \
    .set("spark.driver.extraJavaOptions", "-Dlog4j.logger.org.apache.hadoop.util.NativeCodeLoader=ERROR")
sc = SparkContext(conf = conf)
# Set log level to ERROR to suppress Spark warnings
sc.setLogLevel("ERROR")

# Load documents (one per line).
# Read file with Python to avoid Java security issues with newer Java versions
script_dir = os.path.dirname(os.path.abspath(__file__))
tsv_path = os.path.join(script_dir, "subset-small.tsv")
with open(tsv_path, "r", encoding="utf-8") as f:
    lines = [line.rstrip('\n\r') for line in f.readlines()]
# Partition the data to avoid large task warnings (split into multiple partitions)
# Use more partitions to keep task size below 1000 KiB recommendation
num_partitions = max(16, len(lines) // 250)  # More partitions = smaller tasks
rawData = sc.parallelize(lines, numSlices=num_partitions)
fields = rawData.map(lambda x: x.split("\t")).filter(lambda x: len(x) >= 4)

# Create a combined RDD with (doc_name, doc_text, doc_words) to maintain alignment
docInfo = fields.map(lambda x: (x[1], x[3], x[3].split(" ")))

# Store document names, texts, and word lists separately (in same order)
documentNames = docInfo.map(lambda x: x[0])
documentTexts = docInfo.map(lambda x: x[1])
documents = docInfo.map(lambda x: x[2])

# Now hash the words in each document to their term frequencies:
hashingTF = HashingTF(100000)  #100K hash buckets just to save some memory
tf = hashingTF.transform(documents)

# At this point we have an RDD of sparse vectors representing each document,
# where each value maps to the term frequency of each unique hash value.

# Let's compute the TF*IDF of each term in each document:
# minDocFreq=1 means a term must appear in at least 1 document (no filtering)
# This ensures "Gettysburg" is included even if it only appears in a few documents
tf.cache()
idf = IDF(minDocFreq=1).fit(tf)
tfidf = idf.transform(tf)

# Now we have an RDD of sparse vectors, where each value is the TFxIDF
# of each unique hash value for each document.

# I happen to know that the article for "Abraham Lincoln" is in our data
# set, so let's search for "Gettysburg" (Lincoln gave a famous speech there):

# First, let's figure out what hash value "Gettysburg" maps to by finding the
# index a sparse vector from HashingTF gives us back:
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# Collect all data together to ensure alignment
# Zip TF-IDF vectors back with document names and texts
tfidfWithInfo = tfidf.zip(documentNames).zip(documentTexts).collect()

# Process in Python: find documents containing "Gettysburg" and get their TF-IDF scores
# Also get the tokenized documents to check word counts
docInfoList = docInfo.collect()
allResults = []
for idx, item in enumerate(tfidfWithInfo):
    tfidf_vec, doc_name = item[0]
    doc_text = item[1]
    # Check if document actually contains "Gettysburg" (case-insensitive)
    if "gettysburg" in doc_text.lower():
        # Get TF-IDF score for the hash value
        score = float(tfidf_vec[gettysburgHashValue])
        # If score is 0, try counting the word in tokenized form and computing manually
        if score == 0.0 and idx < len(docInfoList):
            words = docInfoList[idx][2]  # Get tokenized words
            gettysburg_count = sum(1 for w in words if w.lower() == "gettysburg")
            if gettysburg_count > 0:
                # Use a simple TF score if hash doesn't match (fallback)
                score = float(gettysburg_count) / len(words) if len(words) > 0 else 0.0
        allResults.append((score, doc_name, doc_text))

resultCount = len(allResults)

if resultCount > 0:
    # Sort by score descending and get the best result
    allResults.sort(key=lambda x: -x[0])
    bestResult = allResults[0]
    print("Best document for Gettysburg is:")
    print(f"Score: {bestResult[0]:.4f}, Document: {bestResult[1]}")
    
    # Also show top 5 results that actually contain "Gettysburg"
    topResults = allResults[:5]
    print(f"\nTop 5 documents containing 'Gettysburg' (out of {resultCount}):")
    for score, doc_name, doc_text in topResults:
        print(f"  Score: {score:.4f}, Document: {doc_name}")
else:
    print("No documents found containing 'Gettysburg'")
    # Debug: collect all results to see what we got
    allScores = []
    for item in tfidfWithInfo:
        tfidf_vec, doc_name = item[0]
        doc_text = item[1]
        score = float(tfidf_vec[gettysburgHashValue])
        if score > 0.0:
            allScores.append((score, doc_name, doc_text))
    
    allScores.sort(key=lambda x: -x[0])
    print(f"\nTop 10 results by TF-IDF score (checking for 'Gettysburg' in text):")
    for score, doc_name, doc_text in allScores[:10]:
        has_word = "✓" if "gettysburg" in doc_text.lower() else "✗"
        print(f"  Score: {score:.4f}, Document: {doc_name} {has_word}")
