import csv
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

# Log file path
log_file = '/private/var/log/apache2/access_log'

# Empty list for required data
cleaned_logs = []

# Read data and push to list
with open(log_file, 'r') as f:
    logs = f.readlines()

for log in logs:
    parts = log.split()
    try:
        ip = parts[0]
        timestamp = parts[3].strip('[') + ' ' + parts[4].strip(']')
        url = parts[6]
        cleaned_logs.append((ip, timestamp, url))
    except IndexError:
        continue

# Save the cleaned data to a CSV file
csv_file = 'cleaned_logs.csv'

with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['IP Address', 'Timestamp', 'URL'])
    csv_writer.writerows(cleaned_logs)

print(f'Cleaned logs have been saved to {csv_file}')

# Load cleaned logs into DataFrame
df = pd.DataFrame(cleaned_logs, columns=['IP Address', 'Timestamp', 'URL'])

# Vectorize the URL data
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))  # Adjust parameters
X = vectorizer.fit_transform(df['URL'])

# Convert vectors to numpy array
vectors = X.toarray().astype(np.float32)

# Check for non-zero vectors
if np.count_nonzero(vectors) == 0:
    raise ValueError("All vectors are zero vectors. Check the input data and vectorizer parameters.")

# Initialize Pinecone
api_key = 'c6c168dd-b1de-423d-b1ed-48a98cbdc0ad'
pc = Pinecone(api_key=api_key)

# Create Pinecone index if it does not exist
index_name = 'logdata'  # Ensure the index name is lowercase and valid

# Check if index exists
indexes = pc.list_indexes().names()
if index_name in indexes:
    # If the index exists, delete it
    pc.delete_index(index_name)

# Create new index with correct dimension
dimension = vectors.shape[1]
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'  # Updated region based on the free plan's available regions
    )
)

index = pc.Index(index_name)

# Upsert vectors to Pinecone index
ids = [str(i) for i in range(len(vectors))]
index.upsert(vectors=zip(ids, vectors))

# Query the index with the first vector
result = index.query(queries=vectors[:1], top_k=2)
print(result)
